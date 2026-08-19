#!/bin/sh
set -eu

find_bun() {
    if command -v bun >/dev/null 2>&1; then
        command -v bun
        return
    fi
    if [ -x "${BUN_INSTALL:-$HOME/.bun}/bin/bun" ]; then
        printf '%s\n' "${BUN_INSTALL:-$HOME/.bun}/bin/bun"
        return
    fi
    return 1
}

temporary_directory=

cleanup() {
    if [ -n "$temporary_directory" ]; then
        rm -f "$temporary_directory/install.ts" "$temporary_directory/bun-install.sh"
        rmdir "$temporary_directory" 2>/dev/null || true
    fi
}
trap cleanup EXIT HUP INT TERM

make_temporary_directory() {
    if [ -z "$temporary_directory" ]; then
        temporary_directory=$(mktemp -d "${TMPDIR:-/tmp}/code-indexing-mcp.XXXXXX")
    fi
}

download() {
    download_url=$1
    download_target=$2
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$download_url" -o "$download_target"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$download_target" "$download_url"
    else
        echo "Error: curl or wget is required to download the installer." >&2
        exit 1
    fi
}

install_bun() {
    if ! command -v bash >/dev/null 2>&1; then
        echo "Error: Bash is required to install Bun automatically." >&2
        exit 1
    fi
    make_temporary_directory
    bun_installer="$temporary_directory/bun-install.sh"
    download "${CODE_INDEXING_MCP_BUN_INSTALLER_URL:-https://bun.sh/install}" "$bun_installer"
    echo "Bun was not found; installing it from https://bun.sh ..."
    BUN_INSTALL=${BUN_INSTALL:-$HOME/.bun} bash "$bun_installer"
}

run_installer() {
    installer_bun=$1
    installer_file=$2
    shift 2
    if [ -t 0 ]; then
        "$installer_bun" "$installer_file" "$@"
    elif [ -r /dev/tty ] && ( : </dev/tty ) 2>/dev/null; then
        "$installer_bun" "$installer_file" "$@" </dev/tty
    else
        "$installer_bun" "$installer_file" "$@"
    fi
}

if bun_command=$(find_bun); then
    :
else
    install_bun
    if ! bun_command=$(find_bun); then
        echo "Error: Bun installation completed but its executable was not found." >&2
        exit 1
    fi
fi

case "$0" in
    install.sh | */install.sh)
        script_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
        if [ -f "$script_directory/packages/installer/src/bootstrap.ts" ]; then
            run_installer "$bun_command" "$script_directory/packages/installer/src/bootstrap.ts" "$@"
            exit $?
        fi
        ;;
esac

installer_url=${CODE_INDEXING_MCP_INSTALLER_URL:-https://raw.githubusercontent.com/MarcinHamiga/code-indexing-mcp/main/ts/packages/installer/src/bootstrap.ts}
make_temporary_directory
installer_file="$temporary_directory/install.ts"

download "$installer_url" "$installer_file"

run_installer "$bun_command" "$installer_file" "$@"
