#!/usr/bin/env bash
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

myshuf() {
  perl -MList::Util=shuffle -e 'print shuffle(<>);' "$@";
}

normalize_text() {
  tr '[:upper:]' '[:lower:]' | sed -e 's/^/__label__/g' | \
    sed -e "s/'/ ' /g" -e 's/"//g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' \
        -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
        -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' | tr -s " " | myshuf
}

RESULTDIR=result
DATADIR=data

mkdir -p "${RESULTDIR}"
mkdir -p "${DATADIR}"

if [ ! -f "${DATADIR}/dbpedia.train" ]
then
  wget -c "https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz" -O "${DATADIR}/dbpedia_csv.tar.gz"
  tar -xzvf "${DATADIR}/dbpedia_csv.tar.gz" -C "${DATADIR}"
  cat "${DATADIR}/dbpedia_csv/train.csv" | normalize_text > "${DATADIR}/dbpedia.train"
  cat "${DATADIR}/dbpedia_csv/test.csv" | normalize_text > "${DATADIR}/dbpedia.test"
fi

