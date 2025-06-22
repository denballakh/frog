{ pkgs, lib, config, inputs, ... }:
let
  pkgs-unstable = import inputs.nixpkgs-unstable { system = pkgs.stdenv.system; };
  py_base = pkgs.python313;
  py = py_base.override {
    enableOptimizations = true;
    reproducibleBuild = false;
    enableLTO = true;
    self = py_base;
  };
in
{
  # https://devenv.sh/basics/

  # https://devenv.sh/packages/
  packages = [
    pkgs.git
    pkgs.basedpyright
    (py.withPackages (python-pkgs: [
      python-pkgs.types-tqdm
      python-pkgs.tqdm
      python-pkgs.mypy
      python-pkgs.black
    ]))

  ];

  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    package = py;
  };

  languages.nix.enable = true;

  # git-hooks.hooks = {
  #   alejandra.enable = true;
  #   black.enable = true;
  #   # mypy.enable = true;
  #   # pyright.enable = true;
  #   typos.enable = true;
  # };

}
