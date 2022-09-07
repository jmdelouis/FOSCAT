
tab=['test2Dplot.py','test2D.py','testHealpix.py','testHplot.py','testHealpix_mpi.py']
fout=open('src/foscat/build_demo.py','w')
fout.write('def genDemo():')
for fname in tab:
     fout.write('\t\n\t\n\t#=============================\t\n\t\n')
     fout.write('\tf=open("%s","w")\n\n'%(fname))
     f=open('src/tests/%s'%(fname))
     for name in f:
         fout.write('\tf.write("'+name[:-1]+'\\n")\n')
     f.close()
fout.write('\tf.close()\t\n')
fout.write('#======= END OF DEMO ========\n')
fout.close()

