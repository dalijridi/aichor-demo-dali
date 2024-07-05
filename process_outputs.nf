#!/usr/bin/env nextflow

process firstProcess {
  output:
  path 'aichorfoo.txt', optional: true

  script:
  '''
  echo Hello world from AIchor > aichorfoo.txt
  '''
}

process secondProcess {
  input:
  path '*'
  script:
  '''
  cat aichorfoo.txt
  '''
}

workflow {
  firstProcess | secondProcess
}
