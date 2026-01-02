#include "lsm_vec_db.h"

#include <cmath>
#include <exception>
#include <fstream>

#include "lsm_vec_index.h"

namespace lsm_vec
{
namespace {
constexpr char kDefaultVectorFileName[] = "vector.log";
constexpr char kDefaultLogFileName[] = "lsm_vec_db.log";
} // namespace

LSMVecDB::LSMVecDB(const LSMVecDBOptions& options,
                   std::unique_ptr<LSMVec> index,
                   std::unique_ptr<std::ostream> log_stream)
    : options_(options),
      index_(std::move(index)),
      log_stream_(std::move(log_stream))
{
}

Status LSMVecDB::Open(const std::string& path,
                      const LSMVecDBOptions& opts,
                      std::unique_ptr<LSMVecDB>* db)
{
    if (!db) {
        return Status::InvalidArgument("db output must not be null");
    }
    if (opts.dim <= 0) {
        return Status::InvalidArgument("vector dimension must be positive");
    }
    if (opts.metric != DistanceMetric::kL2) {
        return Status::NotSupported("only L2 distance is supported");
    }

    LSMVecDBOptions normalized_opts = opts;
    if (normalized_opts.vector_file_path.empty()) {
        normalized_opts.vector_file_path = path + "/" + kDefaultVectorFileName;
    }

    std::string log_path = opts.log_file_path.empty()
        ? path + "/" + kDefaultLogFileName
        : opts.log_file_path;
    auto log_stream = std::make_unique<std::ofstream>(log_path, std::ios::out);
    if (!static_cast<std::ofstream*>(log_stream.get())->is_open()) {
        return Status::IOError("failed to open log file");
    }

    auto index = std::make_unique<LSMVec>(path, normalized_opts, *log_stream);

    *db = std::unique_ptr<LSMVecDB>(
        new LSMVecDB(normalized_opts, std::move(index), std::move(log_stream)));
    return Status::OK();
}

Status LSMVecDB::ValidateVector(Span<float> vec) const
{
    if (vec.size() != static_cast<size_t>(options_.dim)) {
        return Status::InvalidArgument("vector dimension mismatch");
    }
    return Status::OK();
}

Status LSMVecDB::EnsureMetricSupported() const
{
    if (options_.metric != DistanceMetric::kL2) {
        return Status::NotSupported("only L2 distance is supported");
    }
    return Status::OK();
}

float LSMVecDB::ComputeDistance(Span<float> a, Span<float> b) const
{
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

Status LSMVecDB::Insert(node_id_t id, Span<float> vec)
{
    Status metric_status = EnsureMetricSupported();
    if (!metric_status.ok()) {
        return metric_status;
    }
    Status vec_status = ValidateVector(vec);
    if (!vec_status.ok()) {
        return vec_status;
    }

    try {
        std::vector<float> data(vec.begin(), vec.end());
        index_->insertNode(id, data);
        deleted_ids_.erase(id);
    } catch (const std::exception& ex) {
        return Status::IOError(ex.what());
    }

    return Status::OK();
}

Status LSMVecDB::Update(node_id_t id, Span<float> vec)
{
    Status vec_status = ValidateVector(vec);
    if (!vec_status.ok()) {
        return vec_status;
    }

    try {
        std::vector<float> data(vec.begin(), vec.end());
        index_->vector_storage_->storeVectorToDisk(id, data);
        deleted_ids_.erase(id);
    } catch (const std::exception& ex) {
        return Status::IOError(ex.what());
    }

    return Status::OK();
}

Status LSMVecDB::Delete(node_id_t id)
{
    deleted_ids_.insert(id);
    return Status::OK();
}

Status LSMVecDB::Get(node_id_t id, std::vector<float>* vec)
{
    if (!vec) {
        return Status::InvalidArgument("output vector must not be null");
    }
    if (deleted_ids_.count(id) > 0) {
        return Status::NotFound("vector deleted");
    }

    try {
        index_->vector_storage_->readVectorFromDisk(id, *vec);
    } catch (const std::exception& ex) {
        return Status::IOError(ex.what());
    }

    return Status::OK();
}

Status LSMVecDB::SearchKnn(Span<float> query,
                           const SearchOptions& options,
                           std::vector<SearchResult>* out)
{
    if (!out) {
        return Status::InvalidArgument("output results must not be null");
    }
    if (options.k <= 0) {
        return Status::InvalidArgument("k must be positive");
    }
    if (options.k != 1) {
        return Status::NotSupported("only k=1 search is supported");
    }

    Status metric_status = EnsureMetricSupported();
    if (!metric_status.ok()) {
        return metric_status;
    }

    Status vec_status = ValidateVector(query);
    if (!vec_status.ok()) {
        return vec_status;
    }

    std::vector<float> query_vec(query.begin(), query.end());
    node_id_t result = index_->knnSearch(query_vec);
    if (result == k_invalid_node_id) {
        return Status::NotFound("index is empty");
    }
    if (deleted_ids_.count(result) > 0) {
        return Status::NotFound("nearest neighbor deleted");
    }

    std::vector<float> stored;
    Status get_status = Get(result, &stored);
    if (!get_status.ok()) {
        return get_status;
    }

    out->clear();
    out->push_back({result, ComputeDistance(query, Span<float>(stored))});
    return Status::OK();
}
} // namespace lsm_vec
