
import os
HHDB = "/202121511223/wangc/data/UniRef30_2023_02_hhsuite"
Software_path = "/202121511223/wangc/app/"
HHBLITS = Software_path + "hhsuite/hhblits"

fasta_path = '/202121511223/wangc/code/github/GraphSite-master/AF2_single_representation/'
hhm_path = '/202121511223/wangc/code/pdc_model/data/hhm_files/'
for file_name in os.listdir(fasta_path):
    if file_name.endswith('.fasta'):
        os.system("{0} -i {1} -ohhm {2}.hhm -oa3m {2}.a3m -d {3} -v 0 -maxres 40000 -cpu 6 -Z 0 -o {2}.hhr".format(HHBLITS,fasta_path+file_name, hhm_path+file_name[:-4], HHDB))

print("all files finish")
"""
UR90 = "/202121511223/data/uniref90.fasta.gz.1"
HHDB = "/202121511223/data/UniRef30_2023_02_hhsuite"

Software_path = "/202121511223/app/"
PSIBLAST = Software_path + "ncbi-blast-2.14.0+/bin/psiblast"
HHBLITS = Software_path + "hhsuite/hhblits"
def MSA(data_path, ID):
    os.system("{0} -db {1} -num_iterations 3 -num_alignments 1 -num_threads 4 -query {2}{3}.fa -out {2}{3}.bla -out_ascii_pssm {2}{3}.pssm".format(PSIBLAST, UR90, data_path, ID))
    os.system("{0} -i {1}{2}.fa -ohhm {1}{2}.hhm -oa3m {1}{2}.a3m -d {3} -v 0 -maxres 40000 -cpu 6 -Z 0 -o {1}{2}.hhr".format(HHBLITS, data_path, ID, HHDB))
    """

