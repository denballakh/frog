{ pkgs, lib, config, inputs, ... }:
let
  pkgs-unstable = import inputs.nixpkgs-unstable { system = pkgs.stdenv.system; };
in
{
  # https://devenv.sh/basics/

  # https://devenv.sh/packages/
  packages = [
    pkgs.git
    pkgs.basedpyright
    pkgs.python313
    pkgs.python313Packages.types-tqdm
    pkgs.python313Packages.tqdm
    pkgs.python313Packages.mypy
    pkgs.python313Packages.black
  ];

  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    package = pkgs.python313;
  };

  # languages.nix.enable = true;

  # git-hooks.hooks = {
  #   alejandra.enable = true;
  #   black.enable = true;
  #   # mypy.enable = true;
  #   # pyright.enable = true;
  #   typos.enable = true;
  # };

}
