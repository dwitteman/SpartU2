# Shared helpers. Source this file; do not execute it.

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[$(date +%H:%M:%S)] $*"; }

# toml_get FILE SECTION KEY [DEFAULT]
# Minimal TOML reader: flat [section] / key = value only. Strips quotes and
# trailing comments. Dies if the key is missing and no default is given.
toml_get() {
    local file=$1 section=$2 key=$3 val
    val=$(awk -v sec="$section" -v key="$key" '
        /^[[:space:]]*\[/ { gsub(/[][[:space:]]/, ""); cur = $0; next }
        cur == sec {
            line = $0
            sub(/[[:space:]]+#.*$/, "", line)
            if (match(line, "^[[:space:]]*" key "[[:space:]]*=")) {
                v = substr(line, RLENGTH + 1)
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
                gsub(/^"|"$/, "", v)
                print v; exit
            }
        }' "$file")
    if [[ -z $val ]]; then
        [[ $# -ge 4 ]] || die "missing [$section] $key in $file"
        val=$4
    fi
    echo "$val"
}

# calc EXPR -- floating point arithmetic via awk, e.g. calc "0.4e-3 / 2"
calc() { awk "BEGIN { printf \"%.10g\n\", ($*) }"; }

# growth_rate LENGTH NCELLS FIRST_CELL
# Geometric ratio r such that NCELLS cells starting at FIRST_CELL span LENGTH.
# Returns 1 (uniform) if FIRST_CELL is already >= the uniform spacing.
growth_rate() {
    awk -v L="$1" -v n="$2" -v h="$3" 'BEGIN {
        if (h * n >= L) { print 1; exit }
        lo = 1 + 1e-12; hi = 2
        for (i = 0; i < 200; i++) {
            r = 0.5 * (lo + hi)
            if (h * (r^n - 1) / (r - 1) > L) hi = r; else lo = r
        }
        printf "%.8f\n", r
    }'
}
