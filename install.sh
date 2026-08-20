#!/bin/sh
set -eu

runtime=python
expect_runtime_value=false
for argument in "$@"; do
    if [ "$expect_runtime_value" = true ]; then
        runtime=$argument
        expect_runtime_value=false
        continue
    fi
    case "$argument" in
        --runtime)
            expect_runtime_value=true
            ;;
        --runtime=*)
            runtime=${argument#--runtime=}
            ;;
    esac
done

if [ "$expect_runtime_value" = true ]; then
    echo "Error: --runtime requires python or ts." >&2
    exit 1
fi

if [ "$runtime" != "python" ] && [ "$runtime" != "ts" ]; then
    echo "Error: --runtime must be python or ts." >&2
    exit 1
fi

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

if [ "$runtime" = "ts" ]; then
    case "$0" in
        install.sh | */install.sh)
            script_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
            if [ -f "$script_directory/ts/install.sh" ]; then
                exec sh "$script_directory/ts/install.sh" "$@"
            fi
            ;;
    esac

    installer_url=${CODE_INDEXING_MCP_TS_INSTALLER_URL:-https://raw.githubusercontent.com/MarcinHamiga/code-indexing-mcp/main/ts/install.sh}
    temporary_directory=$(mktemp -d "${TMPDIR:-/tmp}/code-indexing-mcp.XXXXXX")
    installer_file="$temporary_directory/install.sh"
    cleanup_ts() {
        rm -f "$installer_file"
        rmdir "$temporary_directory" 2>/dev/null || true
    }
    trap cleanup_ts EXIT HUP INT TERM
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$installer_url" -o "$installer_file"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$installer_file" "$installer_url"
    else
        echo "Error: curl or wget is required to download the installer." >&2
        exit 1
    fi
    sh "$installer_file" "$@"
    exit $?
fi

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
