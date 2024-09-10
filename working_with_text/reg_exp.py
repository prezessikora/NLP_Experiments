import re
import PyPDF2


# Enter your regex pattern here. This may take several tries!


with open('Business_Proposal.pdf','rb') as f:
    pdf_reader = PyPDF2.PdfReader(f)
    page = pdf_reader.pages[1]
    page_two_text = page.extract_text()
    
    pattern = r'[\w]+@[\w]+.\w{3}'
    result = re.findall(pattern, page_two_text)
    for e in result:
        print(e)