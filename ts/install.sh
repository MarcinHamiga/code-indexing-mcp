#!/bin/sh
set -eu

find_bun() {
    if command -v bun >/dev/null 2>&1; then
        command -v bun
        return
    fi
    echo "Error: Bun >= 1.2 is required but was not found in PATH." >&2
    echo "Install it from https://bun.sh" >&2
    exit 1
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

bun_command=$(find_bun)

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
temporary_directory=$(mktemp -d "${TMPDIR:-/tmp}/code-indexing-mcp.XXXXXX")
installer_file="$temporary_directory/install.ts"

cleanup() {
    rm -f "$installer_file"
    rmdir "$temporary_directory" 2>/dev/null || true
}
trap cleanup EXIT HUP INT TERM

if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$installer_url" -o "$installer_file"
elif command -v wget >/dev/null 2>&1; then
    wget -qO "$installer_file" "$installer_url"
else
    echo "Error: curl or wget is required to download the installer." >&2
    exit 1
fi

run_installer "$bun_command" "$installer_file" "$@"
