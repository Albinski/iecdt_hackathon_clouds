#!/bin/bash
#
# Submit your embeddings to the leaderboard.
#
#   ./submit.sh embeddings/test/yourteam.npz
#
# Copies the file into the organisers' drop directory. A poller picks it up
# within a few minutes and writes your result into the feedback directory
# printed below.
#
# The copy goes to `<name>.part` and is renamed, because a rename is atomic: the
# poller never sees a half-written file, so you never get a spurious "truncated
# file" back. There is no token -- the file's owner is who you are, so submit
# from your own JASMIN account.

set -euo pipefail

ROOT=${IECDT_ROOT:-/gws/ssde/j25b/iecdt/modis_hackathon}
DROP=${IECDT_DROP:-$ROOT/submissions/incoming}
FEEDBACK=${IECDT_FEEDBACK:-$ROOT/submissions/feedback}
LEADERBOARD=${IECDT_LEADERBOARD:-$ROOT/leaderboard}

FILE=${1:?usage: submit.sh <embeddings.npz>}

if [ ! -f "$FILE" ]; then
    echo "error: $FILE does not exist" >&2
    exit 1
fi
case "$FILE" in
    *.npz) ;;
    *) echo "error: $FILE is not a .npz. Submit the file embed.py wrote," >&2
       echo "       not a checkpoint and not an archive of one." >&2
       exit 1 ;;
esac
if [ ! -d "$DROP" ]; then
    echo "error: $DROP does not exist. Check the path announced at the" >&2
    echo "       kickoff, or set IECDT_DROP." >&2
    exit 1
fi

NAME="$(whoami)-$(date -u +%Y%m%dT%H%M%SZ).npz"

cp "$FILE" "$DROP/$NAME.part"
# The organisers have to be able to read it, and they are not in your primary
# group: a umask of 077 would otherwise write a file only you can open, and the
# submission would be refused for that and nothing else.
chmod 644 "$DROP/$NAME.part"
mv "$DROP/$NAME.part" "$DROP/$NAME"

SIZE=$(du -h "$FILE" | cut -f1)
echo "Submitted $FILE ($SIZE) as $NAME"
echo
echo "Your result appears in $FEEDBACK/<your team>.md within a few minutes,"
echo "and the standings are in $LEADERBOARD/leaderboard.md -- read them with"
echo "  uv run python -m iecdt_hackathon.print_leaderboard"
