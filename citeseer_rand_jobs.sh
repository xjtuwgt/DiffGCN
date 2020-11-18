#!/bin/sh
JOBS_PATH=citeseer_jobs
LOGS_PATH=logs
for ENTRY in "${JOBS_PATH}"/*.sh; do
  chmod +x $ENTRY
  FILE_NAME="$(basename "$ENTRY")"
  echo $FILE_NAME
  /mnt/cephfs2/asr/users/ming.tu/software/kaldi/egs/wsj/s5/utils/queue.pl -q g2.q -l gpu=1 $LOGS_PATH/$FILE_NAME.log $ENTRY &
  sleep 3
done