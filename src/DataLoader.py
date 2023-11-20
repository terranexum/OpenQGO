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

     encapsulating_start_marks = list(encapsulating_marks.keys())
     encapsulating_end_marks = list(encapsulating_marks.values())

     orig_file = None
     geojson_file = None 

     
     #ASSUMPTIONS MADE

     #EACH LINE REPRESENTS A NEW NODE

     #Pattern of Encapsulating, Separating, Encapsulating, Different Separating

     def prepareFile(self):
          
          i = 0

          #check each line of data file
          for line in self.orig_file:
               
               #for testing, stop after the first line
               if i == 1: break

               pattern_arr = self.createPatternArray(line)
               desired_pattern_arr = self.getDesiredPatternArr(pattern_arr)
               
               print(desired_pattern_arr)

               #self.assignFieldsAndValues(desired_pattern_arr=desired_pattern_arr, line=line)


               i += 1
               


     def createPatternArray(self, line):

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
               if char in self.encapsulating_start_marks:
                    
                    
                    #because quotations are the same for starting and ending marks, has to have a separate case sadly
                    if char == "\"" and encapsulating_stack[-1][0] == "\"":
                         start_idx = encapsulating_stack[-1][1]

                         encapsulating_stack.pop()
                         encapsulating_indices.append(("\"", start_idx, index))

                         pattern_arr.append(("enc", encapsulating_indices[-1]))

                    else:
                         encapsulating_stack.append((char, index))
                    
               #if we've gotten to an end encapsulating mark
               elif char in self.encapsulating_end_marks:
                    
                    #check last item we added in stack
                    last_item = encapsulating_stack[-1]

                    #if the end mark we found matches up with the starting mark, we can take it out
                    if self.encapsulating_marks[last_item[0]] == char:

                         encapsulating_stack.pop()
                         #store indices within the line that these encapsulating marks cover
                         encapsulating_indices.append((last_item[0], last_item[1], index))
                         pattern_arr.append(("enc", encapsulating_indices[-1]))


               elif char in self.separating_marks:
                    
                    separating_marks.append((char, index))
                    pattern_arr.append(("sep", separating_marks[-1]))


               index += 1

          return pattern_arr
               

     def getDesiredPatternArr(self, pattern_arr):
          
          #desired pattern : enc, sep, enc, sep, enc, sep, enc, sep...
          reached_desired_pattern = False

          des_pattern_arr = []

          i = 0

          while i < len(pattern_arr) - 1:

               marker = pattern_arr[i][0]
               next_marker = pattern_arr[i+1][0]

               #repeating marker
               if marker == next_marker:

                    have_collapsed = False
                    first_bound_index = pattern_arr[i][1][1]

                    if marker == "sep":
                         des_pattern_arr.append(pattern_arr[i])
                         first_bound_index = pattern_arr[i+1][1][1]

                    j = i + 1

                    while (not have_collapsed) and (j < len(pattern_arr)):

                         first_encap_idx = pattern_arr[j][1][1]

                         if first_encap_idx < first_bound_index: 

                              des_pattern_arr.append(pattern_arr[j])
                              have_collapsed = True

                         j += 1

                    i = j
                    
               else:

                    des_pattern_arr.append(pattern_arr[i])
                    i += 1
                    
          return des_pattern_arr

     def assignFieldsAndValues(self, line : str, desired_pattern_arr):

          kvp_idx = 0

          for pattern_marker in desired_pattern_arr:

               if pattern_marker[0] == "enc":
                    
                    first_encap_idx = pattern_marker[1][1]
                    second_encap_idx = pattern_marker[1][2]

                    print(type(second_encap_idx))

                    if kvp_idx % 2 == 0:

                         self.fields.append(line[first_encap_idx:second_encap_idx + 1])
                    
                    else:

                         self.values.append(line[first_encap_idx:second_encap_idx + 1])

                    kvp_idx += 1
                    

          for value in self.values:

               print(value)

          return 

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