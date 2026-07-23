#!/bin/sh
set -eu

find_python() {
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return
    fi
    if command -v python >/dev/null 2>&1; then
        command -v python
        return
    fi
    echo "Error: Python 3 is required but was not found in PATH." >&2
    exit 1
}

run_installer() {
    installer_python=$1
    installer_file=$2
    shift 2
    if [ -t 0 ]; then
        "$installer_python" "$installer_file" "$@"
    elif [ -r /dev/tty ] && ( : </dev/tty ) 2>/dev/null; then
        "$installer_python" "$installer_file" "$@" </dev/tty
    else
        "$installer_python" "$installer_file" "$@"
    fi
}

python_command=$(find_python)

case "$0" in
    install.sh | */install.sh)
        script_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
        if [ -f "$script_directory/install.py" ]; then
            run_installer "$python_command" "$script_directory/install.py" "$@"
            exit $?
        fi
        ;;
esac

installer_url=${CODE_INDEXING_MCP_INSTALLER_URL:-https://raw.githubusercontent.com/MarcinHamiga/code-indexing-mcp/main/install.py}
temporary_directory=$(mktemp -d "${TMPDIR:-/tmp}/code-indexing-mcp.XXXXXX")
installer_file="$temporary_directory/install.py"

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

run_installer "$python_command" "$installer_file" "$@"
