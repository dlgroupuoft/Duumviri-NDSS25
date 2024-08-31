    # -*- coding: utf-8 -*-
"""
@author: CJR
"""
import code
import time
import functools
import os
import json

from urllib.parse import urlparse
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from BlockExtraction import BlockExtraction
from BlockVo import BlockVo
from SeparatorDetection import SeparatorDetection
from SeparatorVo import SeparatorVo
from SeparatorWeight import SeparatorWeight
from ContentStructureConstruction import ContentStructureConstruction
from ImageOut import ImageOut
from CssBox import CssBox
from DomNode import DomNode


class VipsPageSlimmer:
    domjs = """
function toJSON(node) {
  node = node || this; 
  var obj = {
    nodeType: node.nodeType
  };
  if (node.tagName) {
    obj.tagName = node.tagName.toLowerCase();
  } else
  if (node.nodeName) {
    obj.nodeName = node.nodeName;
  }
  if (node.nodeValue) {
    obj.nodeValue = node.nodeValue;
  } 
  if (node.nodeType == 1){
    obj.visual_cues = getCSS(node);
  }
  
  var attrs = node.attributes;
  if (attrs) {
    var length = attrs.length;
    var arr = obj.attributes = new Array(length);
    for (var i = 0; i < length; i++) {
      attr = attrs[i];
    }
  }
  var childNodes = node.childNodes;
  if (childNodes) {
    length = childNodes.length;
    arr = obj.childNodes = new Array(length);
    for (i = 0; i < length; i++) {
      if(childNodes[i].tagName != 'script'){
        arr[i] = toJSON(childNodes[i]);
      }    
    }
  }
  return obj;
}

function getCSS(node){
    var visual_cues = {};
    style =  window.getComputedStyle(node);
    visual_cues["bounds"] = node.getBoundingClientRect();
    visual_cues["font-size"] = style.getPropertyValue("font-size");
    visual_cues["font-weight"] = style.getPropertyValue("font-weight");
    visual_cues["background-color"] = style.getPropertyValue("background-color");
    visual_cues["display"] = style.getPropertyValue("display");
    visual_cues["visibility"] = style.getPropertyValue("visibility");
    visual_cues["text"] = node.innerText
    visual_cues["className"] = node.className
    return visual_cues;
}


function toDOM(obj) {
  if (typeof obj == 'string') {
    obj = JSON.parse(obj);
  }
  var node, nodeType = obj.nodeType;
  switch (nodeType) {
    case 1: //ELEMENT_NODE
      node = document.createElement(obj.tagName);
      var attributes = obj.attributes || [];
      for (var i = 0, len = attributes.length; i < len; i++) {
        var attr = attributes[i];
        node.setAttribute(attr[0], attr[1]);
      }
      break;
    case 3: //TEXT_NODE
      node = document.createTextNode(obj.nodeValue);
      break;
    case 8: //COMMENT_NODE
      node = document.createComment(obj.nodeValue);
      break;
    case 9: //DOCUMENT_NODE
      node = document.implementation.createDocument();
      break;
    case 10: //DOCUMENT_TYPE_NODE
      node = document.implementation.createDocumentType(obj.nodeName);
      break;
    case 11: //DOCUMENT_FRAGMENT_NODE
      node = document.createDocumentFragment();
      break;
    default:
      return node;
  }
  if (nodeType == 1 || nodeType == 11) {
    var childNodes = obj.childNodes || [];
    for (i = 0, len = childNodes.length; i < len; i++) {
      node.appendChild(toDOM(childNodes[i]));
    }
  }
  return node;
}

    """
    PDoc = 1
    Round = 1
    browser = None
    count = 0
    html = None
    cssBoxList = dict()
    nodeList = []
    count3 = 0
    
    def __init__(self, driver, w, h):
        self.w = w
        self.h = h
        self.getDomTree(driver)
        
    
    def service(self):
        #print('-----------------------------Block Extraction------------------------------------')
        be = BlockExtraction()
        block = be.service(self.nodeList)
        blockList = be.blockList
        ret = None
        i = 0
        #print("%s ndoes" % (len(self.nodeList)) )
        while self.checkDoc(blockList) and i<self.Round:
            #print ("blockList.size::", len(blockList))
            # self.imgOut.outBlock(blockList, self.fileName,i)
            ret = blockList

            # ImageOut.outText(self.fileName, blockList, i)                    
            #print("-----------------------------Separator Detection---------------------------------"+str(i))
            sd = SeparatorDetection(self.w, self.h)
            verticalList = []
            verticalList.extend(sd.service(blockList, SeparatorVo.TYPE_VERTICAL))
            # self.imgOut.outSeparator(verticalList, self.fileName, '_vertica_', i)
            
            horizList = []
            horizList.extend(sd.service(blockList, SeparatorVo.TYPE_HORIZ))
            # self.imgOut.outSeparator(horizList, self.fileName,'_horizontal_', i)
            
            #print("-----------------------Setting Weights for Separators----------------------------"+str(i))
            hrList = be.hrList
            sw = SeparatorWeight(self.nodeList)
            sw.service(horizList, hrList)
            sw.service(verticalList, hrList)
            
            #print("-----------------------Content Structure Construction----------------------------"+str(i))
            sepList = []
            sepList.extend(horizList)
            sepList.extend(verticalList)
            sepList.sort(key=functools.cmp_to_key(Vips.sepCompare))
            tempList = blockList
            csc = ContentStructureConstruction()
            csc.service(sepList)
            BlockVo.refreshBlock(block)
            blockList.clear()
            be.filList(block)
            blockList = be.blockList
        
            for newBlock in blockList:
                for oldBlock in tempList:
                    if newBlock.id == oldBlock.id:
                        blockList.remove(newBlock)
                        break
            
            #ImageOut.outText(self.fileName, blockList , 'test')
            i+=1
        return ret
                             
    def checkDoc(self, blocks):
        for blockVo in blocks:
            if blockVo.Doc < self.PDoc:
                return True
        return False

    def toDOM(self, obj, parentNode=None):
        if (isinstance(obj,str)):
            json_obj = json.loads(obj)  #use json lib to load our json string
        else:
            json_obj = obj
        nodeType = json_obj['nodeType']
        node = DomNode(nodeType)
        if nodeType == 1: #ELEMENT NODE
            node.createElement(json_obj['tagName'])
            attributes = json_obj['attributes']
            if attributes != None:
                node.setAttributes(attributes)
            visual_cues = json_obj['visual_cues']
            if visual_cues != None:
                node.setVisual_cues(visual_cues)
        elif nodeType == 3:
            node.createTextNode(json_obj['nodeValue'], parentNode)
            if node.parentNode != None:
                visual_cues = node.parentNode.visual_cues
                if visual_cues != None:
                    node.setVisual_cues(visual_cues)    
        else:
            return node
            
        self.nodeList.append(node)
        if nodeType == 1:
            childNodes = json_obj['childNodes']
            for i in range(0, len(childNodes)):
                if(childNodes[i]['nodeType'] == 1):
                    node.appendChild(self.toDOM(childNodes[i],node))
                if childNodes[i]['nodeType'] == 3:
                    try:
                        if not childNodes[i]['nodeValue'].isspace():
                            node.appendChild(self.toDOM(childNodes[i],node))
                    except KeyError:
                        pass
                        #print('abnormal text node')
                    
        return node

        
    def getDomTree(self, driver):
        #time.sleep(3)
        #read in our DOM js file as string
        jscript = self.domjs
        #add additional javascript code to run our DOM js's toJSON method
        jscript += '\nreturn JSON.stringify(toJSON(document.getElementsByTagName("BODY")[0]));'
        #finally run the javascript, and wait for it to finish and call the someCallback function.
        x = driver.execute_script(jscript)
        ##print(x)
        self.toDOM(x)
             
        
    def setRound(self,round):
        self.Round = round
        
    @staticmethod
    def sepCompare(sep1, sep2):
        if sep1.compareTo(sep2) < 0:
            return -1
        elif sep1.compareTo(sep2) > 0:
            return 1
        else: return 0    

class Vips:
    PDoc = 1
    Round = 1
    url = None
    fileName = None
    browser = None
    count = 0
    imgOut = None
    html = None
    cssBoxList = dict()
    nodeList = []
    count3 = 0
    
    def __init__(self, urlStr):
        self.setUrl(urlStr)
        self.setDriver()
        self.imgOut = ImageOut()
        self.imgOut.outImg(self.browser, self.url, self.fileName)
        self.getDomTree()
    
                
    def service(self):
        #print('-----------------------------Block Extraction------------------------------------')
        be = BlockExtraction()
        block = be.service(self.nodeList)
        blockList = be.blockList
        i = 0
        #print("%s ndoes" % (len(self.nodeList)) )
        while self.checkDoc(blockList) and i<self.Round:
            #print ("blockList.size::", len(blockList))
            self.imgOut.outBlock(blockList, self.fileName,i)
            ImageOut.outText(self.fileName, blockList, i)                    
            #print("-----------------------------Separator Detection---------------------------------"+str(i))
            sd = SeparatorDetection(self.browser.get_window_size()['width'], self.browser.get_window_size()['height'])
            verticalList = []
            verticalList.extend(sd.service(blockList, SeparatorVo.TYPE_VERTICAL))
            self.imgOut.outSeparator(verticalList, self.fileName, '_vertica_', i)
            
            horizList = []
            horizList.extend(sd.service(blockList, SeparatorVo.TYPE_HORIZ))
            self.imgOut.outSeparator(horizList, self.fileName,'_horizontal_', i)
            
            #print("-----------------------Setting Weights for Separators----------------------------"+str(i))
            hrList = be.hrList
            sw = SeparatorWeight(self.nodeList)
            sw.service(horizList, hrList)
            sw.service(verticalList, hrList)
            
            #print("-----------------------Content Structure Construction----------------------------"+str(i))
            sepList = []
            sepList.extend(horizList)
            sepList.extend(verticalList)
            sepList.sort(key=functools.cmp_to_key(Vips.sepCompare))
            tempList = blockList
            csc = ContentStructureConstruction()
            csc.service(sepList)
            BlockVo.refreshBlock(block)
            blockList.clear()
            be.filList(block)
            blockList = be.blockList
        
            for newBlock in blockList:
                for oldBlock in tempList:
                    if newBlock.id == oldBlock.id:
                        blockList.remove(newBlock)
                        break
            
            #ImageOut.outText(self.fileName, blockList , 'test')
            i+=1
              
        self.browser.quit()

    def checkDoc(self, blocks):
        for blockVo in blocks:
            if blockVo.Doc < self.PDoc:
                return True
        return False
    
    def setUrl(self, urlStr):
        try:
            if urlStr.startswith('https://') or urlStr.startswith('http://'):
                self.url = urlStr
            else:
                self.url = 'http://' + urlStr                
            parse_object = urlparse(self.url)
            newpath = r'Screenshots/'+ parse_object.netloc +'_'+str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')) +'/'
            self.fileName = newpath + parse_object.netloc
            os.makedirs(newpath)
        except (TypeError, AttributeError):
            pass
            #print ("Invalid address: " + str(urlStr))
      
    def setDriver(self):
        # CHROME_PATH = r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"  # chrome path
        #CHROME_PATH = r"C:\Users\Tarc\AppData\Local\Google\Chrome\Application\chrome.exe"
        # CHROMEDRIVER_PATH = r"C:\chromedriver_win32\chromedriver.exe" # driver path
        #ADBLOCK_PATH = r"C:\Users\Hard-\AppData\Local\Google\Chrome\User Data\Default\Local Extension Settings\cfhdojbkjhnklbpkdaibdccddilifddb"
        """
        WIDTH = 1080
        HEIGHT = 1920
        PIXEL_RATIO = 1.0
        UA = 'Mozilla/5.0 (Linux; Android 4.1.1; GT-N7100 Build/JRO03C) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/35.0.1916.138 Mobile Safari/537.36 T7/6.3'
        
        mobileEmulation = {"deviceMetrics": {"width": WIDTH, "height": HEIGHT, "pixelRatio": PIXEL_RATIO}, "userAgent": UA}
        """
        CHROME_PATH = './chromium/ungoogled-chromium_111.0.5563.65-1.1.AppImage'		
        CHROMEDRIVER_PATH = './chromium/chromedriver_v111'	

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_experimental_option('prefs', {'intl.accept_languages': 'en,en_US'})
        chrome_options.add_argument('--disable-gpu')
        #chrome_options.add_experimental_option('mobileEmulation', mobileEmulation)
        #chrome_options.add_argument('--disable-popup-blocking')
        #chrome_options.add_argument('load-extention='+ADBLOCK_PATH)
        #chrome_options.add_extension('Adblock-Plus_v3.1.crx')
        chrome_options.binary_location = CHROME_PATH        
        self.browser = webdriver.Chrome(executable_path=CHROMEDRIVER_PATH, chrome_options=chrome_options)
        

    def toDOM(self, obj, parentNode=None):
        if (isinstance(obj,str)):
            json_obj = json.loads(obj)  #use json lib to load our json string
        else:
            json_obj = obj
        nodeType = json_obj['nodeType']
        node = DomNode(nodeType)
        if nodeType == 1: #ELEMENT NODE
            node.createElement(json_obj['tagName'])
            attributes = json_obj['attributes']
            if attributes != None:
                node.setAttributes(attributes)
            visual_cues = json_obj['visual_cues']
            if visual_cues != None:
                node.setVisual_cues(visual_cues)
        elif nodeType == 3:
            node.createTextNode(json_obj['nodeValue'], parentNode)
            if node.parentNode != None:
                visual_cues = node.parentNode.visual_cues
                if visual_cues != None:
                    node.setVisual_cues(visual_cues)    
        else:
            return node
            
        self.nodeList.append(node)
        if nodeType == 1:
            childNodes = json_obj['childNodes']
            for i in range(0, len(childNodes)):
                if(childNodes[i]['nodeType'] == 1):
                    node.appendChild(self.toDOM(childNodes[i],node))
                if childNodes[i]['nodeType'] == 3:
                    try:
                        if not childNodes[i]['nodeValue'].isspace():
                            node.appendChild(self.toDOM(childNodes[i],node))
                    except KeyError:
                        pass
                        #print('abnormal text node')
                    
        return node
    
    def get_url(self):
        self.browser.get(self.url)
        
    def getDomTree(self):
        #time.sleep(3)
        #read in our DOM js file as string
        file = open("dom.js", 'r')
        jscript = file.read()
        #add additional javascript code to run our DOM js's toJSON method
        jscript += '\nreturn JSON.stringify(toJSON(document.getElementsByTagName("BODY")[0]));'
        #finally run the javascript, and wait for it to finish and call the someCallback function.
        x = self.browser.execute_script(jscript)
        ##print(x)
        self.toDOM(x)
             
        
    def setRound(self,round):
        self.Round = round
        
    @staticmethod
    def sepCompare(sep1, sep2):
        if sep1.compareTo(sep2) < 0:
            return -1
        elif sep1.compareTo(sep2) > 0:
            return 1
        else: return 0

        
    
        
   
        
        
    
    