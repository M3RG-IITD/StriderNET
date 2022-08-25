
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

class Lammps_Traj():
    def __init__(self,filepath :str):

        self.Filepath=filepath
        file=open(self.Filepath,'r+')
        prev_line=[]
        Frame=dict()
        self.Traj=[]
        k=0
        flag=False
        temp=0
        for line in file:
            if("TIMESTEP" in prev_line):
                Frame.update(TIMESTEP=int(line))
            elif("NUMBER OF ATOMS" in prev_line):
                Frame.update(N=int(line))
            elif("BOX BOUNDS" in prev_line or flag==True):
                k+=1
                if("BOX BOUNDS" in prev_line):
                    flag=True
                if(k==1):
                    words=self.break_line_to_words(line,' ')
                    Frame.update(xlo=float(words[0]),xhi=float(words[1]))#,xy=float(words[2]))
                elif(k==2):
                    words=self.break_line_to_words(line,' ')
                    Frame.update(ylo=float(words[0]),yhi=float(words[1]))#,xz=float(words[2]))
                elif(k==3):
                    words=self.break_line_to_words(line,' ')
                    Frame.update(zlo=float(words[0]),zhi=float(words[1]))#,zx=float(words[2]))
                
                if(k>2):
                    flag=False
            elif("ATOMS" in prev_line):
                words=self.break_line_to_words(prev_line,' ')
                myheader=words[2:]
                data=pd.read_table(self.Filepath,sep='\s+',skiprows=temp,nrows=Frame.get('N'),header=None,names=myheader)
                Frame.update(Data=data)
                self.Traj+=[dict(Frame)]
                Frame.clear()
            temp+=1
            prev_line=line


    def break_line_to_words(self,line,delim :str =' '):
        """To give list of words in a line with the delimiter as given"""
        res_wordlist=[]
        n=len(line)
        i=0
        word=""
        while(i<n):
            if(line[i]==' ' or line[i]=="\t" or i==n-1):
                if(word!=""):
                    res_wordlist+=[word]
                    word=""
            else:
                word+=line[i]
            i+=1
        return res_wordlist

    def getTraj(self):
        return self.Traj
  