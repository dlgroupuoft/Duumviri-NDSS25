# -*- coding: utf-8 -*-
"""
@author: CJR
"""

'''
main function
'''
import Vips
from urllib.parse import unquote

def main():
    vips = Vips.Vips(unquote("https://www.samsung.com/", encoding="utf-8"))
    vips.setRound(20)
    vips.service()
    
main()
