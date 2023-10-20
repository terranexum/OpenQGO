import json
import os

import pandas
import geopandas

#stack
from collections import deque

class QGOFile: 

     file_path = ""

     fields = []
     values = []

     encapsulating_marks = {
          
          '{' : '}',
          '[' : ']',
          '(' : ')',
          '<' : '>',
          "\"" : "\""
     }

     separating_marks = [

          ',',
          ':',
          

     ]

     orig_file = None
     geojson_file = None 

     
     #ASSUMPTIONS MADE

     #EACH LINE REPRESENTS A NEW NODE

     #Pattern of Encapsulating, Separating, Encapsulating, Different Separating

     def prepareFile(self):
          

          encapsulating_start_marks = list(self.encapsulating_marks.keys())
          encapsulating_end_marks = list(self.encapsulating_marks.values())

          i = 0

          

          #check each line of data file
          for line in self.orig_file:
               
               if i == 1: break

               #create stack to store encapsulating indices
               #first in, last out -- the first encapsulating marks we find will be the last to get out
               encapsulating_stack = deque()

               #encapsulating indices store tuples of indices(representing the range of indices that encapsulating marks cover)
               encapsulating_indices = []

               #separating indices are just one value of what the separating mark index is
               separating_marks = []

               index = 0

               pattern_arr = []

               

               #iterate over characters
               for char in line:
                    
                    #check if character starts an encapsulating sequence
                    if char in encapsulating_start_marks:
                         
                         
                         #because quotations are the same for starting and ending marks, has to have a separate case sadly
                         if char == "\"" and encapsulating_stack[-1][0] == "\"":
                              start_idx = encapsulating_stack[-1][1]

                              encapsulating_stack.pop()
                              encapsulating_indices.append((start_idx, index, "\""))

                              pattern_arr.append(("enc", encapsulating_indices[-1]))

                         else:
                              encapsulating_stack.append((char, index))
                         
                    #if we've gotten to an end encapsulating mark
                    elif char in encapsulating_end_marks:
                         
                         #check last item we added in stack
                         last_item = encapsulating_stack[-1]

                         #if the end mark we found matches up with the starting mark, we can take it out
                         if self.encapsulating_marks[last_item[0]] == char:

                              encapsulating_stack.pop()

                              #store indices within the line that these encapsulating marks cover
                              encapsulating_indices.append((last_item[1], index, last_item[0]))
                              pattern_arr.append(("enc", encapsulating_indices[-1]))

                    #
                    # stack : ['{', '"', ]
                    #
                    #
                    #

                    elif char in self.separating_marks:
                         
                         
                         separating_marks.append((char, index))
                         pattern_arr.append(("sep", separating_marks[-1]))


                    index += 1


               #for j in range(len(pattern_arr)):


               i += 1

               

     def convertTxtToJSON(self):

          fieldValueDict = {}

          for line in self.orig_file:
               

               split : list(str) = line.strip().split()
               print(split)
          
          pass

     def convertJSONToGeoJSON():
          
          pass
     
     def limitData():

          return
     
     #tell the user what data is numeric and what is not
     def getNumericData():

          return
     
     #represent non-numeric data as numeric data
     def createCategoricalData():

          return
     
     #change units of dataset
     def applyScale():

          return

     #filter data 
     def filterData():

          return

     #export GeoJSON file so user can download and use
     def exportGEOJSON():


          return 

     #def multiSplit(a):

# def removeUnnecesssaryMarks():




     

     def __init__(self, file_path : str) -> None:

               self.file_path = file_path
               self.orig_file = open(self.file_path, "r")

               #self.fields = fields

               name , ext = os.path.splitext(file_path)
               
               #self.convertTxtToGeoJSON()
          #
          #     if ext == ".geojson":
          #          geojson_file = self.orig_file
          #     
          #     elif ext == ".txt":
          #          
          #          geojson_file = self.convertTxtToGeoJSON()

          #     elif ext == ".json":
          #          
          #          geojson_file = self.convertJSONToGeoJSON()

               pass
     

qgoFile = QGOFile("C:\\Users\\Anish J\\Desktop\OpenQGO\src\\ELEC.txt")

qgoFile.prepareFile()