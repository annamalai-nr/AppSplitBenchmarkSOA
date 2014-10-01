import sys

def main():
	Lines = [Line.strip() for Line in open(sys.argv[1],'r').readlines()]

	LabelsTxtLocation = str(sys.argv[2]) #currently "/home/annamalai/WATemp/ForExpAppSplitAlone100/ApkToolDecompOp/"

	Apps = []
	Accus = []
	for Line in Lines:
	    App=Line.split(" ")[0].strip()
	    Accu=Line.split(" ")[1].strip()
	    Apps.append(App)
	    Accus.append(Accu)
	
	NoOfLines = []

	for App in Apps:
		FName = "/home/annamalai/WATemp/ForExpAppSplitAlone100/ApkToolDecompOp/" + App +"/smali/Labels.txt"
 		NoOfLine=len(open(FName,'r').readlines())
 		NoOfLines.append(NoOfLine)

 	for Index, App in enumerate(Apps):
 		print NoOfLines[Index]
 		print Accus[Index]






	
if __name__ == '__main__':
		main()