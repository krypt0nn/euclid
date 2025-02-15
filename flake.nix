{
    description = "Euclid: Stupid Natural Language Framework";

    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
        flake-utils.url = "github:numtide/flake-utils";

        rust-overlay = {
            url = "github:oxalica/rust-overlay";
            inputs.nixpkgs.follows = "nixpkgs";
        };
    };

    nixConfig = {
        extra-substituters = [
            "https://cache.nixos.org"
            "https://nix-community.cachix.org"
        ];

        extra-trusted-public-keys = [
            "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
            "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
        ];
    };

    outputs = { self, nixpkgs, flake-utils, rust-overlay }:
        flake-utils.lib.eachDefaultSystem (system:
            let pkgs = import nixpkgs {
                inherit system;

                overlays = [ rust-overlay.overlays.default ];
            };

            in {
                devShells.default = pkgs.mkShell {
                    nativeBuildInputs = with pkgs; [
                        (rust-bin.nightly.latest.default.override {
                            extensions = [ "rust-src" ];
                        })
                        gcc
                    ];
                };
            });
}
