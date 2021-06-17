import layoutparser as lp
import fitz
import cv2
import os
import base64
import warnings
from PIL import Image
import io
import numpy
import requests
import re
from typing import List, Dict, Any
warnings.filterwarnings('ignore')


class GenericParser():
    def __init__(self, 
                 filename: str, 
                 output_doc_name: str,
                 output_doc_path: str, 
                 output_img_path: str,
                 model_name: str, 
                 language: str, 
                 cl: float,
                 method: str = "tesseract | mathpix", 
                 start_page: int = None, 
                 end_page: int = None,):
        """
        Arguments:
             filename: absolute path to the file
             output_doc_name: self-defined name for the output images of the whole pages
             output_doc_path: relative path of the output images of whe whole pages
             output_img_path: relative path of the output figures of current page
             start_page: first page to be processed
             end_page: last page to be processed
             model_name: name of the computer vision model, defined by user 
             method: different approaches to parse images
                 - tesseract: to process non-math documents
                 - mathpix: to process math-based documents
             language: language code to be passed to ocr client.
             cl: confidence level of the model
        """
        
        self.filename = filename
        self.pdf = fitz.open(filename)
        
        self.model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config', 
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", cl],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
        self.ocr_agent = lp.TesseractAgent(languages = language)
        self.model_name = model_name
        self.method = method
        
        self.start_page = start_page
        self.end_page = end_page
    
        
        self.output_doc_name = output_doc_name
        self.output_doc_path = output_doc_path,
        
        self.output_img_path = output_img_path



    def refine_text_block(self,text: str):
        """
        Clean text inside the list
        """
        text = text.strip()
        text = re.sub(r'(?<=\n)[\t ]+', '', text)
        text = re.sub(r'[\t ]+', ' ', text)
        text = re.sub(r'(?<=\w)-\n(?=\w)', '', text)
        text = re.sub(r'“|”', '"', text)
        text = re.sub(r'…', '..', text)
        text = re.sub(r'’', '\'', text)
        text = re.sub(r'—', '-', text)
        text = re.sub(r'\**[•●‣⁃⁌⁍∙○◘◦☙❥❧⦾⦿]', '*', text)
        text = re.sub(r'。', '. ', text)
        
        for s in text:
            if ord(s) < 25 or ord(s) > 127:
                text = text.replace(s,'')

        return text
        
    def refine_latex(self, text: str):
        """
        Convert latext to correct formats, which can be rendered in markdown
        """
        refine_text = re.sub(' +', ' ', text.replace('\\( ',' $').replace(' \\)','$ '))
        return refine_text   
    
    
    def export_page_to_image_(self,page_number: int):
        """
        Take in the page being indicated by the page number,
        then convert page into the image, and output the image 
        to the path defined by users
        """
       
        page = self.pdf[page_number]
        
        zoom_x = 2.0
        zomm_y = 2.0
        
        mat = fitz.Matrix(zoom_x, zomm_y)
        pix = page.get_pixmap(matrix=mat)
        
        pix.writeImage(f'{self.output_doc_path[0]}/{self.output_doc_name}-0{page_number}.png')
        

    
    def convert_page_image(self,page_number: int):
        
        """
        Read the output image, and load it into
        OpenCV library
        """
        path = self.output_doc_path[0]
        image = cv2.imread(f'{self.output_doc_path[0]}/{self.output_doc_name}-0{page_number}.png')
        return image
    
#     def get_image_layout(self, page_number: int):
        
#         image = self.convert_page_image(page_number)
#         layout = self.model.detect(image)
        
#         return layout
        
    
    def to_uri_bdata(self, img):
        """
        Convert the read-in images to the format
        that can be processed by mathpix api
        """
        
        im = Image.fromarray(img.astype("uint8"))
        rawBytes = io.BytesIO()
        im.save(rawBytes, "PNG")
        rawBytes.seek(0)  # return to the start of the file
        b64img = base64.b64encode(rawBytes.read())
        base64_message = b64img.decode('ascii')
        
        return "data:image/jpg;base64," + base64_message
    
    def mathpix(self, img):
        """
        Utilize mathpix api to process images,
        and output the contents into strings which
        might also contains latex string
        """
        src = self.to_uri_bdata(img)
        headers = {
        'app_id': 'josefstrauss_sina_com_314870_8defa1',
        'app_key': '1af27276554898b987df',
        'content-type': 'application/json'}
        
        body = {
        "src": src,
        "formats": ["text", "data", "html"],
        "data_options": {
            "include_asciimath": True,
            "include_latex": True
        }}
        
        data = requests.post("https://api.mathpix.com/v3/text", headers=headers, json=body)
        
        try:
             text = data.json()['text']
        except:
             text = ''
        text = self.refine_latex(text)
        
        return text
        
    def detect_text_blocks(self, page_number: int):
        """
        Detect all blocks inside a image, and return
        the coordinates of text blocks
        """
        image = self.convert_page_image(page_number)
        layout = self.model.detect(image)
        
        text_blocks = lp.Layout([b for b in layout if b.type=='Text' or b.type == 'Title'])
        
        h, w = image.shape[:2]
        left_interval = lp.Interval(0, w/2*1.05, axis='x').put_on_canvas(image)
        
        left_blocks = text_blocks.filter_by(left_interval, center=True)
        left_blocks.sort(key = lambda b:b.coordinates[1])
        
        right_blocks = [b for b in text_blocks if b not in left_blocks]
        right_blocks.sort(key = lambda b:b.coordinates[1])
        
        text_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])
        
        return text_blocks
    
    def detect_figure_blocks(self, page_number):
        '''
        Retrieve all figure areas in side a page image,
        and save both images & image info
        '''
        
        image = self.convert_page_image(page_number)
        layout = self.model.detect(image)
        
        figure_blocks = lp.Layout([b for b in layout if b.type=='Figure'])
        

        figure_info = []
        for i in range(len(figure_blocks)):

            segment_image = (figure_blocks[i]
                       .pad(left=5, right=5, top=5, bottom=5)
                       .crop_image(image))
            img = Image.fromarray(segment_image)
            img.save(f'{self.output_img_path}/{self.output_doc_name}-page{page_number}-0{i}.png')

            figure_block =  figure_blocks[i].to_dict()
            coordinates =[figure_block['x_1'], 
                          figure_block['y_1'],
                          figure_block['x_2'],
                          figure_block['y_2']]
            figure_info.append({'path': f'{self.output_img_path}/{self.output_doc_name}-page{page_number}-0{i}.png',
                                'coordinates': coordinates,
                                'page': page_number,
                                'doc': self.filename})
        return figure_info
            
        
          
    def ocr_by_tesseract(self, page_number: int):
        """
        This function deals with non-math images.
        
        Based on the cooordinates, 
        extract the text from the image,
        clean the text, and returns a list
        of cleaned texts.
        
        
        Returns:
          List[str] -- List of paragraphs
        """
        blocks = self.detect_text_blocks(page_number)
        for block in blocks:
            segment_image = (block
                   .pad(left=5, right=5, top=5, bottom=5)
                   .crop_image(self.convert_page_image(page_number)))
            
            text = self.ocr_agent.detect(segment_image)
            block.set(text=text, inplace=True)
            
            
        text_paras = [txt for txt in blocks.get_texts()]
        clean_text_paras = [self.refine_text_block(txt) for txt in text_paras]
        
        return clean_text_paras
    
    def ocr_by_mathpix(self, page_number):
        """
        This function deals with images having math symbols.
        
        Based on the coordinates, crop the areas, and sent them
        into mathpix, and returns a list of mathpix-processed texts
        
        Returns:
            List[str] -- List of paragraphs
        """
        blocks = self.detect_text_blocks(page_number)
        text_paras = []
        for block in blocks:
            segment_image = (block
                   .pad(left=5, right=5, top=5, bottom=5)
                   .crop_image(self.convert_page_image(page_number)))
            text_paras.append(self.mathpix(segment_image))
            
        return text_paras
        
        
    def ocr(self, page_number: int):
        """
        If defined method name is "tesseract",
        launch the function: "ocr_by_tesseract",
        
        if defined method name is "mathpix",
        launch the function: "ocr_by_mathpix"
        """

        if self.method == "tesseract":
            paras_ocr = self.ocr_by_tesseract(page_number)
            return paras_ocr
        
        elif self.method == "mathpix":
            paras_mathpix = self.ocr_by_mathpix(page_number)
            return paras_mathpix
        
    def parse_figure(self):
        if self.start_page == None:
            self.start_page = 0
            
        if self.end_page == None:
            self.end_page = len(self.pdf)
            
        figures_by_page = []
        for i in range(self.start_page, self.end_page):
            print(i)
            figures_by_page.append({'page': i+1,
                                   'figures': self.detect_figure_blocks(i)})
            
        return figures_by_page
        
        
    def parse_page(self):
        """
        parse multiple pages.
        
        If defined method name is "tesseract",
        it returns: 
            List[str] -- List of text chunks: paragraphs, titles, etc.
            
        If defined method name is "mathpix",
        it retuns:
            List[dict] -- List of dictionaries, where each dictionary
                          contains processed texts, page number, and the filename
        """
        if self.start_page == None:
            self.start_page = 0
        
        if self.end_page == None:
            self.end_page = len(self.pdf)
        
        text_lists = []
        mathpix_results = []
        
        if self.method == 'tesseract':
        
            for i in range(self.start_page, self.end_page):
                self.export_page_to_image_(i)
                text_blocks = self.ocr(i)
                text_lists.append(text_blocks)  
                
            ocr_results = [text_chunk for text_list in text_lists for text_chunk in text_list]
            
            return ocr_results
            
        elif self.method == "mathpix":
            
            for i in range(self.start_page, self.end_page):
                self.export_page_to_image_(i)
                text_blocks = self.ocr(i)
                mathpix_results.append({'text': re.sub(' +', ' ',' '.join(text_blocks)),
                                        'page': i+1,
                                        'doc': self.filename})
        
            return mathpix_results