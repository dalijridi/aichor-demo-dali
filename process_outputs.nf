#!/usr/bin/env nextflow

process foo {
  debug true
  
  output:
  path 'foo.txt', optional: true
  
  script:
  '''
  echo "=== FOO PROCESS STARTED ==="
  echo "Working directory: $(pwd)"
  echo "Hostname: $(hostname)"
  echo "Date: $(date)"
  echo "JAVA_HOME: $JAVA_HOME"
  echo "NXF_HOME: $NXF_HOME"
  echo "PATH: $PATH"
  echo ""
  
  if [[ $(( ( RANDOM % 2 ) )) == 0 ]]; then
    echo "Creating foo.txt..."
    echo "Hello world from foo process" > foo.txt
    echo "foo.txt created successfully"
  else
    echo "Skipping foo.txt creation (random condition)"
  fi
  
  echo "=== FOO PROCESS COMPLETED ==="
  '''
}

process bar {
  debug true
  
  input:
  path '*'
  
  script:
  '''
  echo "=== BAR PROCESS STARTED ==="
  echo "Working directory: $(pwd)"
  echo "Files in directory:"
  ls -la
  echo ""
  
  if [ -f foo.txt ]; then
    echo "Content of foo.txt:"
    cat foo.txt
  else
    echo "foo.txt not found!"
  fi
  
  echo "=== BAR PROCESS COMPLETED ==="
  '''
}

workflow {
  foo | bar
}
