{
  description = "Dev shell with build tools and compression libraries";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          gcc
          gnumake
          cmake
          boost
          jemalloc

          zstd
          snappy
          lz4
          bzip2
          zlib
        ];

        nativeBuildInputs = with pkgs; [
          pkg-config
        ];
      };
    };
}
