rm -rf /tmp/ink-id
git clone --branch develop https://code.cs.uky.edu/seales-research/ink-id.git /tmp/ink-id
sudo singularity build inkid.sif inkid.def

scp inkid.sif lcc:~

ssh otherhost << EOF
  module load ccs/singularity
  rm inkid.overlay && dd if=/dev/zero of=inkid.overlay bs=1M count=500 && mkfs.ext3 inkid.overlay
EOF
