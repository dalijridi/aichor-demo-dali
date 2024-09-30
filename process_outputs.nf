#!/usr/bin/env nextflow

process firstProcess {
  output:
  path 'aichorfoo.txt', optional: true

  script:
  '''
  echo Hello world from AIchor > aichorfoo.txt
  sleep 300
  '''
}

process secondProcess {
  input:
  path '*'
  script:
  '''
  cat aichorfoo.txt
  sleep 500
  '''
}

workflow {
  firstProcess | secondProcess
}
