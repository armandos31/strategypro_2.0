#!/usr/bin/bash

PROCS=`ps -o pid,command`
PID=`echo "$PROCS" | grep streamlit | xargs | cut -d ' ' -f 1`

if [ $PID ]; then
    kill -s SIGINT $PID
    echo "Stopped."
fi;
