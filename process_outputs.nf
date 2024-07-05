#!/usr/bin/env nextflow

process my-first-process {
  output:
  path 'aichor-foo.txt', optional: true

  script:
  '''
  echo Hello world from AIchor > aichor-foo.txt
  '''
}

process my-second-process {
  input:
  path '*'
  script:
  '''
  cat aichor-foo.txt
  '''
}

workflow {
  my-first-process | my-second-process
}
