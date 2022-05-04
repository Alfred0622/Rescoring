#!/bin/bash

a=$1
c=$2

echo ${a}
echo ${c}

/work/jason90255/kaldi/tools/sctk/bin/rover -h /work/jason90255/rescoring/align/dev/align1 ctm -h /work/jason90255/rescoring/align/dev/align2 ctm -h /work/jason90255/rescoring/align/dev/align3 ctm -h /work/jason90255/rescoring/align/dev/align4 ctm -h /work/jason90255/rescoring/align/dev/align5 ctm -h /work/jason90255/rescoring/align/dev/align6 ctm -h /work/jason90255/rescoring/align/dev/align7 ctm -h /work/jason90255/rescoring/align/dev/align8 ctm -h /work/jason90255/rescoring/align/dev/align9 ctm -h /work/jason90255/rescoring/align/dev/align10 ctm -o /work/jason90255/rescoring/align/dev/align.txt -m meth1 -a ${a} -c ${c}
/work/jason90255/kaldi/tools/sctk/bin/rover -h /work/jason90255/rescoring/align/test/align1 ctm -h /work/jason90255/rescoring/align/test/align2 ctm -h /work/jason90255/rescoring/align/test/align3 ctm -h /work/jason90255/rescoring/align/test/align4 ctm -h /work/jason90255/rescoring/align/test/align5 ctm -h /work/jason90255/rescoring/align/test/align6 ctm -h /work/jason90255/rescoring/align/test/align7 ctm -h /work/jason90255/rescoring/align/test/align8 ctm -h /work/jason90255/rescoring/align/test/align9 ctm -h /work/jason90255/rescoring/align/test/align10 ctm -o /work/jason90255/rescoring/align/test/align.txt -m meth1 -a ${a} -c ${c}

echo ${a}
echo ${c}