# Visual Studio Code

The repository includes a local, dependency-free VS Code extension in
`ide/vscode`. It associates `.frog` files with FrogLang and provides TextMate
grammar-based syntax highlighting. It is installed directly from a source
checkout rather than from an extension marketplace. It does not provide a
language server, completion, diagnostics, formatting, or build commands.

## Installation

Run the installation recipe from a source checkout on the machine where the VS Code extension host runs. For a remote or container workspace, this is normally the remote environment rather than the local desktop:

```sh
just vscode-install
```

After confirmation, the recipe creates or updates `~/.vscode/extensions/frog/` as a symbolic link to `ide/vscode` in the current checkout.

Restart VS Code or run **Developer: Reload Window** from the command palette. Opening a `.frog` file should then select the Frog language mode and enable syntax highlighting.

## Updating

Because the installed extension is a symbolic link, changes from the same checkout become available after running **Developer: Reload Window** again. If the checkout moves to another path, rerun `just vscode-install` to update the link target.

## Uninstallation

Remove the symbolic link and reload VS Code:

```sh
unlink "$HOME/.vscode/extensions/frog"
```
