import PyPDF2
              
with open('test.txt') as file:
    lines = file.readlines()
    print(lines)


myfile = open('US_Declaration.pdf',mode='rb')
pdf_reader = PyPDF2.PdfReader(myfile)
print(f'PDF pages: {len(pdf_reader.pages)}')

for p in pdf_reader.pages:
    t = p.extract_text()
    print(t[:20],sep='\n\n')

myfile.close()
    