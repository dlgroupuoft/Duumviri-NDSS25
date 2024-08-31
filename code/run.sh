#/bin/sh
url=${1:-'https://broadsheet.com.au/'}
timestamp=$(date +%s)
domain=`echo "$url" | awk -F/ '{printf $3}'`
fname=${2:-/app/output/$domain-$timestamp}

mkdir -p $fname
echo "run.js -b brave -o $fname -u $url -t 30 --debug debug -i"
built/run.js -b ../../brave-browser/src/out_old/Static/brave -o $fname -u $url -t 30 --debug debug 
