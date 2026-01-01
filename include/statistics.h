#pragma once

#include <cstddef>
#include <ostream>
#include <chrono>

namespace lsm_vec {

class HNSWStats {
public:
    using Clock = std::chrono::high_resolution_clock;

    struct TimerToken {
        bool active  = false;   // false => everything is a no-op
        bool stopped = true;    // true => duration already computed
        Clock::time_point start;
        double duration = 0.0;  // cached duration in seconds
    };

    explicit HNSWStats(bool enabled = false)
        : enabled_(enabled) {}

    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool enabled() const { return enabled_; }

    // ------------------------------------------------------------
    // Generic timing API
    // ------------------------------------------------------------

    // Start a timer. If stats are disabled, this will NOT call Clock::now().
    inline TimerToken startTimer() const {
        TimerToken token;
        if (!enabled_) {
            // inactive token; no time accounting will happen
            return token;
        }
        token.active  = true;
        token.stopped = false;
        token.start   = Clock::now();
        return token;
    }

    // Generic timing function:
    //   - token: TimerToken returned by startTimer()
    //   - time_metric: the double field to accumulate into
    //
    // You can call this multiple times on the same token with different metrics.
    // The duration is computed once and cached in token.duration.
    inline void accumulateTime(TimerToken& token, double& time_metric) const {
        if (!enabled_ || !token.active) return;

        if (!token.stopped) {
            auto end = Clock::now();
            token.duration = std::chrono::duration<double>(end - token.start).count();
            token.stopped  = true;
        }
        time_metric += token.duration;
    }

    // ------------------------------------------------------------
    // Generic counting API
    // ------------------------------------------------------------

    // Generic counting function:
    //   - count: how many operations happened (edges, reads, etc.)
    //   - counter: the size_t field to accumulate into
    inline void addCount(std::size_t count, std::size_t& counter) const {
        if (!enabled_) return;
        counter += count;
    }

    // ------------------------------------------------------------
    // Public metrics (all in lower_snake_case)
    // ------------------------------------------------------------

    // High-level times
    double io_time       = 0.0; // total Aster I/O time
    double indexing_time = 0.0; // total indexing time

    // AsterDB I/O counters and times
    std::size_t io_count            = 0;
    std::size_t read_io_count       = 0;
    std::size_t write_node_io_count = 0;
    std::size_t add_edge_io_count   = 0;
    std::size_t delete_edge_io_count = 0;

    double read_io_time        = 0.0;
    double write_node_io_time  = 0.0;
    double add_edge_io_time    = 0.0;
    double delete_edge_io_time = 0.0;

    std::size_t read_vertex_property_count = 0;
    std::size_t read_edges_count           = 0;
    double      read_vertex_property_time  = 0.0;
    double      read_edges_time            = 0.0;

    // Vector I/O
    std::size_t vec_read_count  = 0;
    double      vec_read_time   = 0.0;
    std::size_t vec_write_count = 0;
    double      vec_write_time  = 0.0;

    // ------------------------------------------------------------
    // Print helper
    // ------------------------------------------------------------

    void print(std::ostream& os) const {
        if (!enabled_) {
            os << "Detailed statistics are disabled." << std::endl;
            return;
        }

        os << "Indexing Time: " << indexing_time << " seconds\n";

        os << "-------graph part------\n";
        os << "Total Aster I/O Operations: " << io_count << "\n";
        os << "Total Aster I/O Time: " << io_time << " seconds\n";
        os << "Read Operations: " << read_io_count
           << ", Time: " << read_io_time << " seconds\n";
        os << "Write Node Operations: " << write_node_io_count
           << ", Time: " << write_node_io_time << " seconds\n";
        os << "Add Edge Operations: " << add_edge_io_count
           << ", Time: " << add_edge_io_time << " seconds\n";
        os << "Delete Edge Operations: " << delete_edge_io_count
           << ", Time: " << delete_edge_io_time << " seconds\n";
        os << "ReadVertexProperty Count: " << read_vertex_property_count
           << ", Time: " << read_vertex_property_time << " seconds\n";
        os << "ReadEdges Count: " << read_edges_count
           << ", Time: " << read_edges_time << " seconds\n";

        os << "-------vector part------\n";
        os << "Total Vector I/O Time: "
           << vec_read_time + vec_write_time << " seconds\n";
        os << "Vector Read Operations: " << vec_read_count
           << ", Time: " << vec_read_time << " seconds\n";
        os << "Vector Write Operations: " << vec_write_count
           << ", Time: " << vec_write_time << " seconds\n";
        os << std::endl;
    }

private:
    bool enabled_ = false;
};

} // namespace ROCKSDB_NAMESPACE
