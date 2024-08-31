from inspect import getframeinfo, stack
import traceback
import requests

import codecs
import pickle
import shutil
import glob
import itertools
import subprocess
import cv2
import numpy as np
import time
import trio
from bs4 import NavigableString
import json
import copy
import seleniumwire
import urllib.robotparser
import os
from niteru import style_similarity, structural_similarity, similarity

from multiprocessing import Pool
import math

import datetime
from readability import Document
from readabilipy import simple_json_from_html_string
from trafilatura import extract as tura_extract

import pandas as pd
from collections import Counter
import sys

import ast
import random
import code
import utils
import traceback
from urllib.parse import urlparse, parse_qs
import re
from tldextract import extract

import feature_extractor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from selenium.webdriver.common.by import By

from scipy.spatial import distance
import networkx as nx
from bs4 import BeautifulSoup


from Vips import VipsPageSlimmer
import selenium
from adblockparser import AdblockRules



def to_odd(num):
	if num % 2 == 1:
		return num
	return max(num - 1, 0)

def apply_mask(res, nond_mask):
	if res.shape[:2] != nond_mask.shape[:2]:
		return res

	if len(nond_mask.shape) > 2:
		nond_mask=np.amax(nond_mask, axis=2)

	nond_mask=cv2.bitwise_not(nond_mask.astype('uint8')).astype('uint8')
	res=cv2.bitwise_and(res, res, mask=nond_mask)
	return res



def remove_h_v_lines(img):
	"""
	TODO
	"""


	img=img.astype(np.uint8)
	ret, img=cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

	h, w=img.shape

	horizontal_size=to_odd(w // 200)
	if (horizontal_size > 0):
		horizontalStructure=cv2.getStructuringElement(
		    cv2.MORPH_RECT, (horizontal_size, 5))
		img=cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontalStructure)


	verticalsize=to_odd(h // 200)
	if (verticalsize > 0):
		verticalStructure=cv2.getStructuringElement(
		    cv2.MORPH_RECT, (5, verticalsize))
		img=cv2.morphologyEx(img, cv2.MORPH_OPEN, verticalStructure)

	return img


def actual_comparison(img1, img2, nond_mask=None, nond_mask2=None):
	if nond_mask is not None and nond_mask.shape[:2] != img1.shape[:2]:
		return None

	h_th, s_th, v_th=20, 30, 30
	bgr_th=20

	img1_hsv=cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
	img2_hsv=cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
	bgr_diff=np.sum(cv2.absdiff(img1, img2), axis=2)
	bgr_res=(bgr_diff > bgr_th).astype('uint8')

	hsv_diff=cv2.absdiff(img1_hsv, img2_hsv)
	h_d, s_d, v_d=cv2.split(hsv_diff)
	h_res=(h_d > h_th).astype('uint8')
	s_res=(s_d > s_th).astype('uint8')
	v_res=(v_d > v_th).astype('uint8')

	res=np.sum(cv2.merge([h_res, s_res, v_res]), axis=2) & bgr_res


	res=remove_h_v_lines(res)

	if nond_mask is not None:
		res=apply_mask(res, nond_mask)

	if nond_mask2 is not None:
		res=apply_mask(res, nond_mask2)


	assert (res.shape[:2] == img1.shape[:2]), "res.shape: %s, img1 "
	return res




tmp_file_path = ''
external_lists1 = utils.ExternalFilters([1, 0, 0, 0, 0, 0])
external_lists2 = utils.ExternalFilters([0, 1, 0, 0, 0, 0])

debugging = 0
debug_fname = '97773_arniesairsoftcouk_0'
debug_i, debug_j = 0, 0

efficient_net_model = None
efficient_net_model_load_time = 0

class OnePageData():
    def __init__(self, driver, window_size=None):

        w, h = -1, -1
        if window_size:
            w, h = window_size['width'], window_size['height']
        self.screenshot, self.window_size = self.get_screenshot(
            driver, w, h, with_mask=False)
        w, h = self.window_size['width'], self.window_size['height']
        self.main_screenshot, self.first_section_screenshot, self.vips_screenshot = self.get_additional_screenshots(
            driver, w, h)
        self.pagegraph = None

        self.requests = copy.deepcopy(driver.requests)

        self.rendered = self.get_rendered_information(driver)

    def element_screenshot(self, driver, tagname):
        try:
            ele = driver.find_element(By.TAG_NAME, tagname)
            if ele.rect['height'] == 0:
                return None
            png = ele.screenshot_as_png
            nparr = np.frombuffer(png, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except selenium.common.exceptions.NoSuchElementException:
            return None
        except:
            traceback.print_exc()

            return None

        return img

    def is_valid_vips_screenshot(self, vips_position):
        x, y, w, h = vips_position
        return (min([w, h]) / max([w, h])) >= 0.5 and min([w, h]) >= 300

    def vips_screenshot(self, driver, w, h):
        vips = VipsPageSlimmer(driver, w, h)
        vips.setRound(10)
        blocks = vips.service()

        max_area = 0
        max_block = None
        if not blocks:
            return None

        for b in blocks:
            if not b.isVisualBlock:
                continue
            block_area = b.width*b.height
            if block_area > max_area:
                max_area = block_area
                max_block = b

        max_block.x = max(int(max_block.x), 0)
        max_block.y = max(int(max_block.y), 0)
        vips_position = [int(max_block.y), int(max_block.y+max_block.height),
                         max(int(max_block.x), 0), int(max_block.x+max_block.width)]

        if self.is_valid_vips_screenshot(vips_position):
            ret = self.screenshot[vips_position[0]
                : vips_position[1], vips_position[2]:vips_position[3]]
            return ret
        return None

    def is_valid_visit(self):
        return self.rendered['body_text']

    def get_additional_screenshots(self, driver, w, h):
        utils.try_set_window_size(driver, w, h)

        vips_screenshot = None

        return self.element_screenshot(driver, 'main'), self.element_screenshot(driver, 'section'), vips_screenshot

    def get_rendered_information(self, driver):
        return utils.FeatureExtractor(driver).get_features()

    def stop_blocking(self, driver):

        del driver.request_interceptor, driver.response_interceptor

    def get_mask(self, driver, first_screenshot):
        time.sleep(30)
        second_screenshot = utils.get_screenshot_as_numpy(driver)
        ret = np.zeros_like(first_screenshot)
        mask = actual_comparison(
            first_screenshot, second_screenshot).astype('uint8')
        if len(mask.shape) > 2:
            mask = np.sum(mask, axis=2)
        mask = np.where(mask > 0, 255, 0).astype('uint8')
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            ret = cv2.rectangle(ret, (x-10, y-10),
                                (x+w+10, y+h+10), (255, 255, 255), -1)
        return ret

    async def handle_execute_context_create(self, receive_channel):
        async with receive_channel:
            async for value in receive_channel:
                print(value)

                self.context = value.context.id_

    async def test_contexts(self, driver):
        with trio.move_on_after(15):
            async with driver.bidi_connection() as connection:
                cdp_session, devtools = connection.session, connection.devtools
                target_create_listener = cdp_session.listen(
                    devtools.runtime.ExecutionContextCreated)
                async with trio.open_nursery() as nursey:
                    nursey.start_soon(
                        self.handle_execute_context_create, target_create_listener)
                    await cdp_session.execute(devtools.runtime.enable())

    def get_screenshot(self, driver, width=-1, height=-1, with_mask=False):

        if width == -1 or height == -1:
            width = 1920
            height = utils.get_max_height(driver)

        utils.try_set_window_size(driver, width, height)

        first_screenshot = utils.get_screenshot_as_numpy(driver)

        window_size = driver.get_window_size()
        if with_mask:
            self.mask = self.get_mask(driver, first_screenshot)

        return first_screenshot, window_size

    def get_pagegraph(self, driver):
        response = driver.execute_cdp_cmd('Page.generatePageGraph', {})
        return response

    def get_pagegraph_externally(self, driver):
        self.tmp_pagegraph_dir = '/tmp/tmp_pagegraph_%d' % random.randint([
                                                                          0, 1000])
        os.makedirs(self.tmp_pagegraph_dir, exist_ok=True)
        assert (driver.current_url)

        import pathlib
        parent_path = pathlib.Path(__file__).parent.resolve()
        cmd = os.path.join(
            parent_path, "./pagegraph-crawl/run.sh %s %s" % (url, self.tmp_pagegraph_dir))

        self.pagegraph_proc = subprocess.Popen(cmd, shell=True, cwd=os.path.join(
            parent_path, './pagegraph-crawl'), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    def read_pagegraph(self):
        return {os.path.basename(p): open(p, 'r').read() for p in glob.glob(os.path.join(self.tmp_pagegraph_dir, '*.graphml'))}

    def wait_pagegraph_externally(self):
        assert (self.pagegraph_proc)
        while (self.pagegraph_proc.wait() is None):
            time.sleep(1)

        if self.pagegraph_proc.returncode == 0:
            self.pagegraph = self.read_pagegraph()
        del self.pagegraph_proc

        shutil.rmtree(self.tmp_pagegraph_dir)


class RobotsInfo():
    def __init__(self, url):
        self.url = url
        self.rp = urllib.robotparser.RobotFileParser()
        self.rp.set_url("%s/robots.txt" % url)
        try:
            self.rp.read(timeout=10)
        except:
            self.rp = None


class PageDelta():
    def __init__(self, save_path, url, idx, rule, rule2=None):
        self.save_path = save_path
        self.url = url
        self.idx = idx
        self.rule = rule
        self.rule2 = rule2
        self.blocked, self.vanilla = None, None
        self.window_size = None
        self.begin = time.time()
        self.retry_count = 2
        if url:
            self.robots_info = RobotsInfo(url)

    def calculate_robot_info_statistics(self, req_urls):
        if self.robots_info.rp is not None:
            self.violations = [
                url for url in req_urls if not self.robots_info.rp.can_fetch('*', url)]
        else:
            self.violations = []

        self.tried_urls = req_urls
        num_violation = len(self.violations)

    def save_blocked(self, driver, cache, retry_func=None):
        begin = time.time()
        self.blocked = None
        self.blocked_rety = []

        for i in range(self.retry_count, 0, -1):
            self.blocked_rety.append(OnePageData(driver, self.window_size))

            if self.window_size is None:
                if self.blocked:
                    self.window_size = self.blocked.window_size
                else:
                    self.window_size = self.blocked_rety[0].window_size

            if i != 1:
                driver = trio.run(retry_func, driver, self.url, cache)

        self.filter = cache
        self.blocked_save_duration = time.time() - begin
        return driver

    def get_pagegraph_externally(self, url):
        self.tmp_pagegraph_dir = '/tmp/tmp_pagegraph_%d' % random.randint(
            0, 1000)
        os.makedirs(self.tmp_pagegraph_dir, exist_ok=True)
        assert (url)

        import pathlib
        parent_path = pathlib.Path(__file__).parent.resolve()
        cmd = os.path.join(
            parent_path, "./pagegraph-crawl/run.sh %s %s" % (url, self.tmp_pagegraph_dir))

        self.pagegraph_proc = subprocess.Popen(cmd, shell=True, cwd=os.path.join(
            parent_path, './pagegraph-crawl'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def read_pagegraph(self):
        try:
            ret = {}
            for p in glob.glob(os.path.join(self.tmp_pagegraph_dir, '*.graphml')):
                try:
                    ret[os.path.basename(p)] = open(p, 'r').read()
                except:
                    print("Error processing %s " % p)
                    traceback.print_exc()

        except:
            traceback.print_exc()

        return ret

    def wait_pagegraph_externally(self):
        assert (self.pagegraph_proc)
        while (self.pagegraph_proc.wait() is None):
            time.sleep(2)

        if self.pagegraph_proc.returncode == 0:
            self.pagegraph = self.read_pagegraph()
        else:
            shutil.rmtree(self.tmp_pagegraph_dir)

            raise Exception('pagegraph error')
        del self.pagegraph_proc
        shutil.rmtree(self.tmp_pagegraph_dir)

    def save_pagegraph(self, driver, save_method=1):
        if save_method == 1:
            self.get_pagegraph_externally(self.url)

            try:
                self.wait_pagegraph_externally()
            except:
                res = self.pagegraph_proc.communicate()
                print("pagegraph %d %s" %
                      (self.pagegraph_proc.returncode, res[1]))
                raise Exception('Pagegraph error')
        elif save_method == 2:
            assert (driver), 'No driver to call Page.generatePageGraph'

    def save(self, driver=None, idx=0, total=0):

        os.makedirs(self.save_path, exist_ok=True)

        utils.dump(self, os.path.join(self.save_path, 'pagedelta.pickle'))

    def save_vanilla(self, driver, cache, retry_func=None):
        begin = time.time()
        self.vanilla = None
        self.vanilla_retry = []

        for i in range(self.retry_count, 0, -1):
            self.vanilla_retry.append(OnePageData(driver, self.window_size))

            if self.window_size is None:
                if self.vanilla:
                    self.window_size = self.vanilla.window_size
                else:
                    self.window_size = self.vanilla_retry[0].window_size
            if i != 1:
                driver = trio.run(retry_func, driver, self.url, cache)

        self.save_pagegraph(driver)
        self.vanilla_save_duration = time.time() - begin
        return driver


class Features():
    def __init__(self):
        pass

    def __repr__(self):
        return ' '.join(["%s:%s" % (k, str(v)) for k, v in vars(self).items()])


class ExtendedPageDelta():
    """
    Convert normal page delta to full page delta
    """

    def __init__(self, page_delta):
        if not page_delta:
            return

        self.begin = time.time()
        self.page_delta = page_delta

        self.to_remove_headers = [
            'Accept-Datetime', 'Date', 'If-Modified-Since', 'If-Unmodified-Since']

        self.features = Features()

        tracker_keywords = ['banner', 'sponser', 'ad', 'ads', 'analytics', 'analytic', 'tag',
                            'metrics', 'metric', 'smetrics', 'stats', 'tracking', 'tracker', 'stat', 'pixel']
        keywords_from_webgraph = ["ad", "ads", "advert", "popup", "banner", "sponsor", "iframe", "googlead", "adsys", "adser", "advertise", "redirect",
                                  "popunder", "punder", "popout", "click", "track", "play", "pop", "prebid", "bid", "pb.min", "affiliate", "ban",
                                  "delivery", "promo", "tag", "zoneid", "siteid", "pageid", "size", "viewid", "zone_id", "google_afc", "google_afs"]
        tracker_keywords = set(tracker_keywords + keywords_from_webgraph)

        self.tracker_keywords_re = [re.compile(
            '\b%s\b' % word) for word in tracker_keywords]

        def calculate_ratio(x): return x[0]/x[1]
        self.common_dim_side = [(300, 250), (336, 280), (250, 250),
                                (200, 200), (180, 150), (125, 125), (250, 360), (580, 400)]
        self.common_dim_side_ratio = list(
            map(calculate_ratio, self.common_dim_side))
        self.common_dim_banner = [(728, 90), (320, 100), (320, 50), (468, 60), (234, 60), (970, 90), (
            970, 415), (970, 250), (980, 120), (930, 180), (750, 300), (750, 200), (750, 100)]
        self.common_dim_banner_ratio = list(
            map(calculate_ratio, self.common_dim_banner))
        self.common_dim_vertical = [
            (300, 600), (120, 600), (120, 240), (160, 600), (300, 1050), (240, 400)]
        self.common_dim_vertical_ratio = list(
            map(calculate_ratio, self.common_dim_vertical))

    def parse_dom(self):

        self.page_delta.vanilla.dom = BeautifulSoup(
            self.page_delta.vanilla.rendered['html'], 'lxml')
        self.page_delta.blocked.dom = BeautifulSoup(
            self.page_delta.blocked.rendered['html'], 'lxml')

    def get_text_similarity(self, t1, t2):
        if t1 is None or not t1 or t2 is None or not t2:
            t1 = '' if t1 is None else t1
            t2 = '' if t2 is None else t2

        if not t1 and not t2:
            return 0

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform([t1, t2])
        arr = X.toarray()
        return distance.cdist([arr[0]], [arr[1]], 'cosine')[0][0]

    def list_of_string_similarity(self, l1, l2):
        if l1 is None or not l1 or l2 is None or not l2:
            l1 = '' if l1 is None else l1
            l2 = '' if l2 is None else l2

        identical_i, identical_j = self.identical_pairs(
            l1, l2, lambda x, y: utils.similarity_score(x, y) >= 0.9)
        o1 = len([l1[i] for i in range(len(l1)) if i not in identical_i])

        return o1

    def count_similarity(self, n1, n2):

        return max(n1 - n2, 0)

    def count_similarity_rendered(self, s1, s2):

        return self.count_similarity(len(s1), len(s2))

    def deduplicate_links(self, ls1, ls2):

        def convert_links(x): return (
            x.attrs['href'] if 'href' in x.attrs else '', x.get_text())
        ls1 = list(map(convert_links, ls1))
        ls2 = list(map(convert_links, ls2))
        def is_identical_link(x, y): return (
            x and utils.similarity_score(x[0], y[0]) > 0.9) or x[1] == y[1]

        identical_i, identical_j = self.identical_pairs(
            ls1, ls2, is_identical_link)
        o1 = len([ls1[i] for i in range(len(ls1)) if i not in identical_i])
        o2 = len([ls2[j] for j in range(len(ls2)) if j not in identical_j])

        return o1

    def deduplicate_script_tags(self, ss1, ss2):
        def convert_ss(x): return (
            x.attrs['src'] if 'src' in x.attrs else '', x.get_text())
        ss1 = list(map(convert_ss, ss1))
        ss2 = list(map(convert_ss, ss2))

        def is_identical_link(x, y): return (x and utils.similarity_score(
            x[0], y[0]) > 0.9) or (y and utils.similarity_score(x[1], y[1]) > 0.9)

        identical_i, identical_j = self.identical_pairs(
            ss1, ss2, is_identical_link)
        o1 = len([ss1[i] for i in range(len(ss1)) if i not in identical_i])
        o2 = len([ss2[j] for j in range(len(ss2)) if j not in identical_j])
        return o1

    def is_sensitive_size(self, x):
        rect = x[1]

        h, w = rect['height'], rect['width']
        if h == 0:
            h = 1
        ratio = w/h
        dim_match = any(map(lambda x: abs(w - x[0]) < 10 and abs(
            h - x[1]) < 10, self.common_dim_side + self.common_dim_banner + self.common_dim_vertical))
        ratio_match = any(map(lambda x: abs(x - ratio) < 0.2, self.common_dim_side_ratio +
                          self.common_dim_banner_ratio + self.common_dim_vertical_ratio))
        return dim_match or ratio_match or h * w < 10

    def dfs(self, node, depth):
        if isinstance(node, NavigableString):
            return depth, 1

        ret = [self.dfs(child, depth+1) for child in node.children]

        max_depth = max([child[0] for child in ret] + [depth])
        total_nodes = sum([child[1] for child in ret])
        return max_depth, total_nodes + 1

    def is_ad_tracking_iframe(self, iframe):
        src, _, innerHTML = iframe
        parsed = BeautifulSoup(innerHTML, 'html.parser')
        num_scripts = len(parsed.find_all('script'))
        num_iframes = len(parsed.find_all('iframe'))
        num_images = len(parsed.find_all('img'))
        text = parsed.get_text()
        max_depth, total_nodes = self.dfs(parsed, 1)

        return len(text) == 0 or 'data-google-query-id' in text

    def deduplicate_rendered_info(self, l1, l2, common_name=""):

        def filter_by_tiny_size(x): return (
            x[1]['height'] * x[1]['width']) <= 10

        def not_tiny_size(x): return (x[1]['height'] * x[1]['width']) > 10

        def filter_by_large_size(x): return (
            x[1]['height'] * x[1]['width']) >= 400

        def filter_large_but_not_sensitive(x): return filter_by_large_size(
            x) and not self.is_sensitive_size(x)

        def filter_sensitive(x): return self.is_sensitive_size(
            x) and not_tiny_size(x)

        sensitive_v, sensitive_b = list(filter(filter_sensitive, l1)), list(
            filter(filter_sensitive, l2))
        small_sized_v, small_sized_b = list(
            filter(filter_by_tiny_size, l1)), list(filter(filter_by_tiny_size, l2))
        sized_v, sized_b = list(filter(filter_large_but_not_sensitive, l1)), list(
            filter(filter_large_but_not_sensitive, l2))

        def is_identical_src_and_size(x, y): return utils.similarity_score(x[0], y[0]) > 0.9 and abs(
            x[1]['height'] - y[1]['height']) <= 20 and abs(x[1]['width'] - y[1]['width']) <= 20

        identical_i, identical_j = self.identical_pairs(
            sensitive_v, sensitive_b, is_identical_src_and_size)
        sensitive_dim_v = len([sensitive_v[i] for i in range(
            len(sensitive_v)) if i not in identical_i])

        identical_i, identical_j = self.identical_pairs(
            small_sized_v, small_sized_b, is_identical_src_and_size)
        small_sized_v_only = len([small_sized_v[i] for i in range(
            len(small_sized_v)) if i not in identical_i])

        identical_i, identical_j = self.identical_pairs(
            sized_v, sized_b, is_identical_src_and_size)
        sized_v_only = len([sized_v[i] for i in range(
            len(sized_v)) if i not in identical_i])

        v_only_ads_frame = []
        if common_name == 'iframes':

            identical_i, identical_j = self.identical_pairs(
                l1, l2, is_identical_src_and_size)
            l1_v_only = [l1[i] for i in range(len(l1)) if i not in identical_i]
            v_only_ads_frame = len(
                list(map(self.is_ad_tracking_iframe, l1_v_only)))

        return small_sized_v_only, sized_v_only, sensitive_dim_v, v_only_ads_frame

    def tura_extract_wrapper(self, html):
        res = tura_extract(html)
        if not res:
            return ""
        return res.lower().replace('advertisement', "")

    def build_appearnce_features(self):
        self.features.vips_screenshot_similaritiy, self.features.main_screenshot_similaritiy, self.features.first_section_screenshot_similarity, self.features.cormier_screenshot_similaritiy = 0, 0, 0, 0

        global efficient_net_model, efficient_net_model_load_time
        if efficient_net_model is None:
            efficient_net_model = feature_extractor.load_model(
                feature_vector=True)
            efficient_net_model_load_time += 1

        feature_vectors = [feature_extractor.get_feature_vectors(efficient_net_model, [
                                                                 self.page_delta.vanilla.screenshot, self.page_delta.blocked.screenshot])]
        self.features.feature_vector_similaritiy = distance.cdist(
            [feature_vectors[0][0]], [feature_vectors[0][1]], 'cosine')[0][0]

        if self.features.feature_vector_similaritiy and self.page_delta.vanilla.vips_screenshot is not None and self.page_delta.blocked.vips_screenshot is not None:
            feature_vectors = [feature_extractor.get_feature_vectors(efficient_net_model, [
                                                                     self.page_delta.vanilla.vips_screenshot, self.page_delta.blocked.vips_screenshot])]
            self.features.vips_screenshot_similaritiy = distance.cdist(
                [feature_vectors[0][0]], [feature_vectors[0][1]], 'cosine')[0][0]

        if self.page_delta.vanilla.main_screenshot is not None and self.page_delta.blocked.main_screenshot is not None and self.features.feature_vector_similaritiy:
            feature_vectors = [feature_extractor.get_feature_vectors(efficient_net_model, [
                                                                     self.page_delta.vanilla.main_screenshot, self.page_delta.blocked.main_screenshot])]
            self.features.main_screenshot_similaritiy = distance.cdist(
                [feature_vectors[0][0]], [feature_vectors[0][1]], 'cosine')[0][0]

        if self.page_delta.vanilla.first_section_screenshot is not None and self.page_delta.blocked.first_section_screenshot is not None and self.features.feature_vector_similaritiy:
            feature_vectors = [feature_extractor.get_feature_vectors(efficient_net_model, [
                                                                     self.page_delta.vanilla.first_section_screenshot, self.page_delta.blocked.first_section_screenshot])]
            self.features.first_section_screenshot_similarity = distance.cdist(
                [feature_vectors[0][0]], [feature_vectors[0][1]], 'cosine')[0][0]

        fonts_v = self.page_delta.vanilla.rendered['fonts']
        fonts_b = self.page_delta.blocked.rendered['fonts']
        colors_v = [s.replace(' ', '_')
                    for s in self.page_delta.vanilla.rendered['colors']]
        colors_b = [s.replace(' ', '_')
                    for s in self.page_delta.blocked.rendered['colors']]

        self.features.text_similarity = self.get_text_similarity(
            self.page_delta.vanilla.rendered['body_text'], self.page_delta.blocked.rendered['body_text'])

        self.features.readability_text_similarity = self.get_text_similarity(self.tura_extract_wrapper(
            self.page_delta.vanilla.rendered['html']), self.tura_extract_wrapper(self.page_delta.blocked.rendered['html']))
        self.features.style_similarity = style_similarity(
            self.page_delta.vanilla.rendered['html'], self.page_delta.blocked.rendered['html'])
        self.features.structure_similarity = structural_similarity(
            self.page_delta.vanilla.rendered['html'], self.page_delta.blocked.rendered['html'])
        self.features.html_similarity = similarity(
            self.page_delta.vanilla.rendered['html'], self.page_delta.blocked.rendered['html'])

        self.features.font_v = self.list_of_string_similarity(fonts_v, fonts_b)
        self.features.color_v = self.list_of_string_similarity(
            colors_v, colors_b)
        try:
            self.features.height_diff = max(
                self.page_delta.vanilla.rendered['highest_pixel'] - self.page_delta.blocked.rendered['highest_pixel'], 0)
        except:
            self.features.height_diff = 0

        self.features.canvas_diff = self.count_similarity(len(self.page_delta.vanilla.dom.select(
            'canvas')), len(self.page_delta.blocked.dom.select('canvas')))
        self.features.audio_diff = self.count_similarity(len(self.page_delta.vanilla.dom.select(
            'audio')), len(self.page_delta.blocked.dom.select('audio')))

        self.features.button_diff = self.count_similarity(len(self.page_delta.vanilla.dom.select(
            'button')), len(self.page_delta.blocked.dom.select('button')))
        self.features.input_diff = self.count_similarity(len(self.page_delta.vanilla.dom.select(
            'input')), len(self.page_delta.blocked.dom.select('input')))
        self.features.span_diff = self.count_similarity(len(self.page_delta.vanilla.dom.select(
            'span')), len(self.page_delta.blocked.dom.select('span')))

        self.features.unloaded_diff = self.count_similarity(len(
            self.page_delta.vanilla.rendered['cdp_network'][2]), len(self.page_delta.blocked.rendered['cdp_network'][2]))
        self.features.css_files_diff = self.count_similarity(len(
            self.page_delta.vanilla.rendered['cdp_network'][3]), len(self.page_delta.blocked.rendered['cdp_network'][3]))
        if 'load_time' in self.page_delta.vanilla.rendered and 'load_time' in self.page_delta.blocked.rendered:
            self.features.load_time_diff = self.page_delta.vanilla.rendered[
                'load_time'] - self.page_delta.blocked.rendered['load_time']

        self.features.videos_diff_small, self.features.videos_diff_large, self.features.video_sensitive_size, _ = self.deduplicate_rendered_info(
            self.page_delta.vanilla.rendered['videos'], self.page_delta.blocked.rendered['videos'], 'videos')

        self.features.images_diff_small, self.features.images_diff_large, self.features.images_sensitive_size, _ = self.deduplicate_rendered_info(
            self.page_delta.vanilla.rendered['images'], self.page_delta.blocked.rendered['images'], 'images')
        self.features.iframes_diff_small, self.features.iframes_diff_large, self.features.iframes_sensitive_size, self.features.iframes_v_only_ads_frame = self.deduplicate_rendered_info(
            self.page_delta.vanilla.rendered['iframes'], self.page_delta.blocked.rendered['iframes'], 'iframes')

        self.features.links_diff = self.deduplicate_links(
            self.page_delta.vanilla.dom.select('a'), self.page_delta.blocked.dom.select('a'))
        self.features.dom_scripts_diff = self.deduplicate_script_tags(
            self.page_delta.vanilla.dom.select('script'), self.page_delta.blocked.dom.select('script'))

        if debugging:
            code.interact("build_appearnce_features",
                          local=dict(locals(), **globals()))

    def _flatten_storage_values(self, storage_value, res):
        res = []
        if isinstance(storage_value, str):
            try:
                evaled = ast.literal_eval(storage_value)
            except:

                evaled = storage_value

            if isinstance(evaled, str):
                parsed = list(utils.parse_raw_cookie(evaled).values())
                if len(parsed) == 1:
                    res.append(storage_value)
                else:
                    self._flatten_storage_values(parsed, res)
            else:
                self._flatten_storage_values(evaled, res)

        elif isinstance(storage_value, list) or isinstance(storage_value, tuple):
            for item in storage_value:
                self._flatten_storage_values(item, res)
        elif isinstance(storage_value, dict):
            for item in storage_value.values():
                self._flatten_storage_values(item, res)
        elif isinstance(storage_value, int):
            res.append(str(storage_value))

    def flatten_storage_values(self, storage_value):
        res = []
        self._flatten_storage_values(storage_value, res)

        return res

    def extract_storage_values(self, cookies, local_storage, session_storage):
        storage_values = utils.get_cookie_values(cookies) + self.flatten_storage_values(
            local_storage) + self.flatten_storage_values(session_storage)

        storage_values = list(set([v for v in storage_values if
                                   isinstance(v, str) and len(v) > 5]))
        return storage_values

    def prepare_one_graph(self):
        """
        Locate the blocked script nodes and their edges 
        """
        blocked_requests = self.page_delta.filter.get_session_blocked_reqs()
        blocked_urls = [req.url for req in blocked_requests]

        graph_string_format = self.v_pagegraph[0][1]

        g = nx.parse_graphml(graph_string_format)

        script_nodes = []
        for n in g.nodes:
            if g.nodes[n]['node type'] == 'script' and 'url' in g.nodes[n] and any([os.path.basename(g.nodes[n]['url']) == os.path.basename(blocked_url) for blocked_url in blocked_urls]):
                script_nodes.append(n)

        out_edge_views = [g.out_edges(
            script_node, data=True, keys=True) for script_node in script_nodes]
        script_edges = []
        for out_edge_view in out_edge_views:
            script_edges += list(out_edge_view)

        script_edges = [(u, g.nodes[v], k, data)
                        for u, v, k, data in script_edges]

        return script_edges

    def get_tracking_apis(self, counts):
        tracking_apis1 = ["Document.cookie", "DOMPluginArray.length", "Document.referrer", "Navigator.javaEnabled", "Document.setCookie", "DOMPlugin.name", "Screen.height", "DOMPluginArray.pluginData", "Screen.width", "Document.domain", "Location.url", "Location.host", "Location.hostname", "Document.createDocumentFragment", "Storage.getItem", "HTMLInputElement.value", "Document.title", "HTMLInputElement.maxLength", "Screen.colorDepth", "HTMLInputElement.setValue", "Location.protocol",
                          "Document.webkitHidden", "Location.href", "HTMLElement.setInnerText", "Storage.setItem", "DOMWindow.innerWidth", "Navigator.language", "Element.hasAttribute", "Location.search", "Node.getElementsByName", "Navigator.cookieEnabled", "Document.defaultView", "HTMLImageElement.setHeight", "Navigator.vendor", "HTMLImageElement.setWidth", "Document.createComment", "DOMPluginArray.item", "Element.webkitMatchesSelector", "Location.hash", "HTMLTextAreaElement.value"]
        tracking_apis2 = ['cookie', 'storage', 'img.load', 'WebGL', 'Date', 'Navigator', 'Location', 'timeout', 'canvas',
                          'todataurl', "getImageData", "measureText", "font", "fillText", "strokeText", "fillStyle", "strokeStyle", "save", "restore"]
        sensitive_elements = ['img', 'iframe']

        tracking_apis1 = list(map(lambda x: x.lower(), tracking_apis1))
        tracking_apis2 = list(map(lambda x: x.lower(), tracking_apis2))

        total = 0
        tracking_apis = {}
        for api_name, count in counts.items():

            if api_name in tracking_apis1 or any([tracking_api in api_name for tracking_api in tracking_apis2]) or any([api_name.startswith(sensitive_element) for sensitive_element in sensitive_elements]):
                total += count
            tracking_apis[api_name] = count

        return total, tracking_apis

    def get_connectivity_features(self, script_nodes, g):
        connectivity_features_total = []

        for node in script_nodes:
            try:
                in_degree = g.in_degree(node)
                out_degree = g.out_degree(node)
                in_out_degree = in_degree + out_degree

                ancestor_list = nx.ancestors(g, node)
                ancestors = len(ancestor_list)
                descendants = len(nx.descendants(g, node))

                closeness_centrality = nx.closeness_centrality(g, node)
                average_degree_connectivity = nx.average_degree_connectivity(g)[
                    in_out_degree]

                try:
                    H = g.copy().to_undirected()
                    eccentricity = nx.eccentricity(H, node)
                except Exception as e:
                    eccentricity = -1

                connectivity_features = [in_degree, out_degree, in_out_degree,
                                         ancestors, descendants, closeness_centrality, average_degree_connectivity,
                                         eccentricity]
                connectivity_features_total.append(connectivity_features)

            except Exception as e:
                traceback.print_exc()

        connectivity_features_total = [
            sum(l) for l in list(zip(*connectivity_features_total))]
        if not connectivity_features_total:
            return [0, 0, 0, 0, 0, 0, 0, 0]
        return connectivity_features_total

    def build_features_on_responses(self):
        """
        This function extracts features about the initiating script 
        """

        if self.page_delta.pagegraph is None:
            return

        pd = self.page_delta
        g = utils.load_g_from_pd(pd)
        if not g:
            return

        blocked_requests = self.page_delta.filter.get_session_blocked_reqs()
        blocked_urls = [req.url for req in blocked_requests]

        def hit_func(url):
            """
            This function is used to filter the resources nodes in pagegraph
            """
            matched = []
            for blocked_url in blocked_urls:
                if utils.fuzzy_match(url, blocked_url):
                    matched.append(blocked_url)

            return matched

        script_nodes = utils.get_script_nodes_from_pg(
            g, hit_func, backwards=False)
        self.build_script_nodes_features(
            script_nodes, g, feature_name_prefix="response_")

    def get_status_code_from_requests(self, url, page_delta):
        requests = set()
        for retry in page_delta.vanilla_retry:
            requests = requests.union(retry.requests)

        found = [req for req in requests if utils.fuzzy_match(url, req.url)]
        if found and 'response' in dir(found[0]) and found[0].response and 'status_code' in dir(found[0].response):
            return found[0].response.status_code
        return 0

    def add_features_from_feature_collector(self, feature_collector, feature_name_prefix=""):
        if not feature_name_prefix.endswith('_'):
            feature_name_prefix += '_'

        for attr, value in feature_collector.__dict__.items():
            setattr(self.features, feature_name_prefix + attr, value)

    def build_script_nodes_features(self, script_nodes, g, feature_name_prefix=""):
        feature_collector = utils.FeatureCollector()

        script_edges = utils.get_script_edges(g, script_nodes, self.page_delta)

        script_edges_flattened = list(itertools.chain(*script_edges))

        feature_collector.graph_total_edges = len(script_edges)
        node_counter = [u for u, v, _, _, _ in script_edges_flattened] + \
            [v for (u, v, _, _, _) in script_edges_flattened]

        node_urls = set([g.nodes[node]['url']
                        for node in script_nodes if 'url' in g.nodes[node]])
        feature_collector.init_script_url_len = sum(
            [len(url) for url in node_urls])

        feature_collector.graph_total_nodes = len(set(node_counter))
        feature_collector.graph_n_by_e = feature_collector.graph_total_nodes / \
            max(feature_collector.graph_total_edges, 0.000001)
        feature_collector.graph_e_by_n = feature_collector.graph_total_edges / \
            max(feature_collector.graph_total_nodes, 0.000001)
        feature_collector.ancestor_ad_keywords = self.get_ad_keywords(
            [g.nodes[script_node] for script_node in script_nodes])
        feature_collector.ancestor_eval_keywords = sum([str(g.nodes[node]).count(
            'eval') for node in script_nodes]) + sum([str(g.nodes[node]).count('function') for node in script_nodes])
        feature_collector.in_degree, feature_collector.out_degree, feature_collector.in_out_degree, feature_collector.ancestors, feature_collector.descendants, feature_collector.closeness_centrality, feature_collector.average_degree_connectivity, feature_collector.eccentricity = self.get_connectivity_features(
            script_nodes, g)

        feature_collector.num_in_redirects, feature_collector.num_out_redirects = 0, 0
        feature_collector.num_create_nodes, feature_collector.num_remove_nodes = 0, 0
        feature_collector.num_DOM_access = 0

        edge_types = set()
        for u, v, k, data, edge_type in script_edges_flattened:
            edge_types.add(data['edge type'])
            if data['edge type'] == 'request start':
                status_code = self.get_status_code_from_requests(
                    g.nodes[v]['url'], self.page_delta)
                if status_code >= 300 and status_code < 400:
                    to_mod = feature_collector.num_out_redirects
                    if edge_type == 'in':
                        to_mod = feature_collector.num_in_redirects
                    to_mod += 1
            elif data['edge type'] in ['create node', 'insert node']:
                feature_collector.num_create_nodes += 1
            elif data['edge type'] in ['delete node', 'remove node']:
                feature_collector.num_remove_nodes += 1
            elif data['edge type'] in ['structure', 'text change', 'set attribute', 'delete attribute']:
                feature_collector.num_DOM_access += 1

        features_stats = utils.extract_graph_features(
            script_edges, g, pd=self.page_delta)

        if features_stats:
            lst_counters = [Counter(stats) for stats, _ in features_stats]
            features_stats = utils.merge_feature_stats(lst_counters)

            feature_collector.graph_num_requests_sent = features_stats['graph_num_requests_sent']
            feature_collector.graph_total_parameters_sent = features_stats[
                'graph_total_parameters_sent']
            feature_collector.graph_avg_parameters_sent = feature_collector.graph_total_parameters_sent / \
                max(feature_collector.graph_num_requests_sent, 1)
            feature_collector.graph_cookie_read_times, feature_collector.graph_cookie_set_times = features_stats[
                'cookie_read_times'], features_stats['cookie_set_times']
            feature_collector.graph_storage_read_times, feature_collector.graph_storage_set_times = features_stats[
                'storage_read_times'], features_stats['storage_set_times']
            feature_collector.storage_read_len = sum(
                list(map(lambda x: len(x[1]), features_stats['read_storage'])))
            feature_collector.storage_write_len = sum(
                list(map(lambda x: len(x[1]), features_stats['write_storage'])))
            feature_collector.cookie_delete_times = features_stats['cookie_delete_times']
            feature_collector.storage_delete_times = features_stats['storage_delete_times']

            def is_third_party(url, fp_domain_list):
                return utils.extract_domain(url) in fp_domain_list
            fp_domain_list = utils.get_top_domain_list(self.page_delta)
            feature_collector.num_tp_storage_read = len(
                [k for k, v, script_url, page_url in features_stats['read_storage'] if is_third_party(script_url, page_url)])
            feature_collector.num_tp_storage_write = len(
                [k for k, v, script_url, page_url in features_stats['write_storage'] if is_third_party(script_url, page_url)])

            feature_collector.graph_num_event_listeners_installed = features_stats[
                'graph_num_event_listeners_installed']
            feature_collector.num_creates = sum(
                features_stats['creates'].values())
            feature_collector.num_listeners = sum(
                features_stats['listeners'].values())
            feature_collector.num_accesses = sum(
                features_stats['bindings'].values()) + sum(features_stats['attributes'].values())

            feature_collector.all_api_counts = features_stats['listeners'] + \
                features_stats['creates'] + \
                features_stats['attributes'] + features_stats['bindings']
            feature_collector.num_tracking_apis, feature_collector.debug_tracking_apis = self.get_tracking_apis(
                feature_collector.all_api_counts)
            try:
                feature_collector.highest_api_calls = max(
                    list(feature_collector.debug_tracking_apis.values()))
            except:
                feature_collector.highest_api_calls = 0

        self.add_features_from_feature_collector(
            feature_collector, feature_name_prefix)

    def build_tua_specific_features(self):
        """
        This function extracts features about the initiating script 
        """

        if self.page_delta.pagegraph is None:
            return

        pd = self.page_delta
        g = utils.load_g_from_pd(pd)
        if not g:
            return

        blocked_requests = self.page_delta.filter.get_session_blocked_reqs()
        blocked_eles = self.page_delta.filter.get_session_blocked_eles()
        blocked_urls = [req.url for req in blocked_requests]

        def hit_func(url):
            """
            This function is used to filter the resources nodes in pagegraph
            """
            matched = []
            for blocked_url in blocked_urls:
                if utils.fuzzy_match(url, blocked_url):
                    matched.append(blocked_url)

            return matched

        if not blocked_urls and not blocked_eles:
            code.interact('no toblock', local=dict(locals(), **globals()))

        script_nodes = utils.get_script_nodes_from_pg(g, hit_func)

        self.build_script_nodes_features(script_nodes, g)

    def storage_diff(self, v1, v2, is_cookies=False):
        if is_cookies:
            def compare_cookies(
                x, y): return x['domain'] == y['domain'] and x['name'] == y['name']

            identical_i, identical_j = self.identical_pairs(
                v1, v2, compare_cookies)
            o1 = len([v1[i] for i in range(len(v1)) if i not in identical_i])

            return o1
        else:
            def compare_storage(x, y): return x[0] == y[0]
            if isinstance(v1, dict):
                v1, v2 = list(v1.items()), list(v2.items())
            else:
                print(v1)
            identical_i, identical_j = self.identical_pairs(
                v1, v2, compare_storage)
            o1 = len([v1[i] for i in range(len(v1)) if i not in identical_i])

            return o1

    def get_listeners_diff_lists(self, l1, l2):
        l1 = list(set(l1))
        l2 = list(set(l2))

        identical_i, identical_j = self.identical_pairs(
            l1, l2, self.is_identical_tuple_listener)
        o1 = len([l1[i] for i in range(len(l1)) if i not in identical_i])
        o2 = len([l2[j] for j in range(len(l2)) if j not in identical_j])

        return o1

    def is_identical_tuple_listener(self, t1, t2):
        assert (len(t1) == 3 and len(t2) == 3)
        return t1[0] == t2[0] and t1[2] == t2[2] and utils.similarity_score(t1[1], t2[1]) >= 0.9

    def filter_listeners2(self, l):
        specific, generic = [], []
        sensitive = []
        for listener in l:
            if len(listener) != 3:
                continue
                code.interact('filter_listeners listener size',
                              local=dict(locals(), **globals()))

            func, ele, event = listener

            ele, event = ele.lower(), event.lower()

            if ele.startswith('<input ') or ele.startswith('<button ') or ele.startswith('<nav ') or ele.startswith('<li ') or ele.startswith('<form ') or ele.startswith('<select ') or ele.startswith('<textarea ') or ele.startswith('<a '):
                specific.append((func, ele, event))
            elif ele.startswith('<div ') or ele.startswith('<body ') or ele.startswith(''):
                generic.append((func, ele, event))

            if (ele.startswith('<script ') and event == 'onerror') or (ele == 'window' and event == 'error') or (ele == 'document' and event == 'submit') or (ele == 'document' and event == 'click') or (ele == 'document' and event == 'auxclick') or (ele == 'document' and event == 'mousedown') or (ele == 'window' and event == 'load'):
                sensitive.append((func, ele, event))

        return specific, generic, sensitive

    def is_critical_ele(self, ele):
        parsed = BeautifulSoup(ele, 'html.parser')

        try:
            all_eles = parsed.find()
            if not all_eles:
                return False

            content = all_eles.content

            if content is None:
                content = ''
            to_check = (content + str(all_eles.attrs)).lower()

            is_submit = parsed.select(
                'input[type="submit"]') or 'submit' in to_check
            is_chat = 'chat' in to_check
            is_comment = 'comment' in to_check or 'reply' in to_check or 'post' in to_check or 'message' in to_check or 'send' in to_check or 'chat' in to_check
            is_captcha = 'captcah' in to_check
            is_download = 'download' in to_check or 'install' in to_check

            return is_submit or is_chat or is_comment or is_captcha or is_download

        except:
            print('%s error finding the html content:%s' %
                  (self.page_delta.save_path, str(parsed)[:100]))

            return False

    def filter_listeners_cdp(self, l):

        specific, generic = [], []
        sensitive = []
        func_related_events = []
        critical_eles = []
        for listener in l:
            assert (isinstance(listener, dict))
            ele, event = listener['node'].lower(), listener['type'].lower()

            if ele.startswith('<input ') or ele.startswith('<button ') or ele.startswith('<nav ') or ele.startswith('<li ') or ele.startswith('<form ') or ele.startswith('<select ') or ele.startswith('<textarea ') or ele.startswith('<a ') or ele.startswith('<video') or ele.startswith('<img') or ele.startswith('<span'):
                specific.append(listener)
            elif ele.startswith('<body ') or ele.startswith('<!doctype'):
                generic.append(listener)

            if event in ['onsubmit', 'oninput', 'onfocus', 'onmouseover', 'onmouseout', 'ondrag', 'ondragend', 'ondragenter', 'ondragleave', 'ondragover', 'ondragstart', 'ondrop']:
                func_related_events.append(listener)

            if self.is_critical_ele(ele):
                critical_eles.append(listener)

            if (ele.startswith('<script ') and 'error' in event) or (ele == 'window' and 'error' in event) or (ele == 'document' and event == 'submit') or (ele == 'document' and
                                                                                                                                                            event == 'click') or (ele == 'document' and event == 'auxclick') or (ele == 'document' and (event == 'mousedown' or event == 'keydown')) or (ele == 'window' and 'load' in event):
                sensitive.append(listener)

        return specific, generic, sensitive, func_related_events, critical_eles

    def add_script_url(self, listeners, script_mapping):
        script_mapping = {
            sp.script_id: sp.url for sp in script_mapping if sp.url}
        ret = []

        for listener in listeners:
            script_id = listener['scriptId']
            if script_id in script_mapping:
                listener['script_url'] = script_mapping[script_id]
            ret.append(listener)
        return ret

    def get_logs_diff(self, lv, lb):

        def filter_log_entry(
            l): return l['level'] == 'SEVERE' and l['source'] != 'network' and 'Adhighlighter' not in l['message']
        lb = list(filter(filter_log_entry, lb))
        lv = list(filter(filter_log_entry, lv))

        return len(lb) - len(lv)

    def get_adhighlighter_diff(self, lv, lb):

        def filter_adhighlighter(l): return 'Adhighlighter:' in l['message']
        lb = list(filter(filter_adhighlighter, lb))
        lv = list(filter(filter_adhighlighter, lv))

        lb, lv = set([l['message'] for l in lb]), set(
            [l['message'] for l in lv])

        return len(lv) - len(lb)

    def build_input_features(self):

        page_specific_listeners_v, generic_listenres_v, sensitive_listeners_v, func_related_listeners_v, critical_ele_listeners_v = self.filter_listeners_cdp(
            self.page_delta.vanilla.rendered['listeners_cdp'])
        page_specific_listeners_b, generic_listenres_b, sensitive_listeners_b, func_related_listeners_b, critical_ele_listeners_b = self.filter_listeners_cdp(
            self.page_delta.blocked.rendered['listeners_cdp'])

        cdp_listeners_v = self.add_script_url(
            self.page_delta.vanilla.rendered['listeners_cdp'], self.page_delta.vanilla.rendered['script_mapping'])
        cdp_listeners_b = self.add_script_url(
            self.page_delta.blocked.rendered['listeners_cdp'], self.page_delta.blocked.rendered['script_mapping'])

        self.features.storage_diff_v = self.storage_diff(
            self.page_delta.vanilla.rendered['local_storage'], self.page_delta.blocked.rendered['local_storage'])
        self.features.session_storage_diff_v = self.storage_diff(
            self.page_delta.vanilla.rendered['session_storage'], self.page_delta.blocked.rendered['session_storage'])

        self.features.cookies_diff_v = self.storage_diff(
            self.page_delta.vanilla.rendered['cookies'], self.page_delta.blocked.rendered['cookies'], is_cookies=True)

        self.features.specific_listeners_v, _ = self.dictionary_diff(
            page_specific_listeners_v, page_specific_listeners_b)
        self.features.generic_listeners_v, _ = self.dictionary_diff(
            generic_listenres_v, generic_listenres_b)
        self.features.sensitive_listeners_v, _ = self.dictionary_diff(
            sensitive_listeners_v, sensitive_listeners_b)
        self.features.func_related_listeners_v, _ = self.dictionary_diff(
            func_related_listeners_v, func_related_listeners_b)
        self.features.critical_ele_listeners_v, _ = self.dictionary_diff(
            critical_ele_listeners_v, critical_ele_listeners_b)

        self.features.logs_diff_b = self.get_logs_diff(
            self.page_delta.vanilla.rendered['log'], self.page_delta.blocked.rendered['log'])
        self.features.ad_highlighter_diff = self.get_adhighlighter_diff(
            self.page_delta.vanilla.rendered['log'], self.page_delta.blocked.rendered['log'])

        if debugging:
            code.interact('build_input_features',
                          local=dict(locals(), **globals()))

    def identical_pairs(self, i1, i2, func):

        identical_i, identical_j = [], []
        for i in range(len(i1)):
            if i in identical_i:
                continue
            dd1 = i1[i]
            for j in range(len(i2)):
                if i in identical_i or j in identical_j:
                    continue

                dd2 = i2[j]
                if func(dd1, dd2):
                    identical_i.append(i)
                    identical_j.append(j)

        return identical_i, identical_j

    def dictionary_diff(self, d1, d2):

        d1 = [dict(t) for t in {tuple(d.items()) for d in d1}]
        d2 = [dict(t) for t in {tuple(d.items()) for d in d2}]

        identical_i, identical_j = self.identical_pairs(
            d1, d2, self.is_identical_dict_listener)
        o1 = len([d1[i] for i in range(len(d1)) if i not in identical_i])
        o2 = len([d2[j] for j in range(len(d2)) if j not in identical_j])

        return o1, o2

    def build_dict_listener_id(self, d, simplified=True):
        if simplified:
            return tuple([d['once'], d['passive'], d['type'], d['useCapture']])
        return tuple([d['once'], d['passive'], d['type'], d['useCapture'], d['columnNumber'], d['lineNumber']])

    def is_identical_dict_listener(self, d1, d2):

        if 'script_url' in d1 and 'script_url' in d2:
            same_url = utils.similarity_score(
                d1['script_url'], d2['script_url'])
            return self.build_dict_listener_id(d1) == self.build_dict_listener_id(d2) and same_url >= 0.9
        elif 'source' in d1 and 'source' in d2:
            souce_similarity = utils.similarity_score(
                d1['source'][:100], d2['source'][:100])
            return self.build_dict_listener_id(d1) == self.build_dict_listener_id(d2) and souce_similarity >= 0.9

        return self.build_dict_listener_id(d1) == self.build_dict_listener_id(d2)

    def get_ad_keywords(self, blocked_reqs):
        if not len(blocked_reqs):
            return 0

        req = blocked_reqs[0]
        if isinstance(req, seleniumwire.request.Request):
            to_search = ' '.join([self.get_req_as_str(req)
                                 for req in blocked_reqs])
        elif isinstance(req, dict):
            to_search = ''
            if 'source' in req:
                to_search += req['source']
            if 'url' in req:
                to_search += req['url']
        else:
            code.interact('get_ad_keywords', local=dict(locals(), **globals()))
        ret = 0

        for pattern in self.tracker_keywords_re:
            ret += len(re.findall(pattern, to_search))
        return ret

    def get_req_as_str(self, req):
        cleaned_headers = {}

        for h, v in req.headers.items():
            if h not in self.to_remove_headers:
                cleaned_headers[h] = v

        ret = ' '.join(
            [str(req.url) + ' ' + str(cleaned_headers) + ' ' + str(req.body)])

        return ret

    def get_storage_values_out(self, storage_values, blocked_requests):
        to_search = [self.get_req_as_str(req) for req in blocked_requests]
        total = sum([to_search.count(str(v)) for v in storage_values])
        return total

    def find_blocked_requests(self, blocked_requests):
        found_reqs = []
        for blocked_req in blocked_requests:
            b_p = urlparse(blocked_req.url)
            for req in self.page_delta.vanilla.requests:
                r_p = urlparse(req.url)
                if utils.similarity_score(b_p.netloc + b_p.path, r_p.netloc + r_p.path) > 0.9 and req.method == blocked_req.method:
                    found_reqs.append(req)
                    break

        return found_reqs

    def count_fp_domain(self, req):
        to_search = self.get_req_as_str(req)
        total = 0
        for fp_domain in self.fp_domain_list:
            total += to_search.count(fp_domain)
        return total

    def get_num_ad_dimension(self, reqs):
        dims = self.common_dim_banner + self.common_dim_side + self.common_dim_vertical
        ret = 0
        for req in reqs:
            found = False
            for w, h in dims:
                if re.search(rf'\b{w}', self.get_req_as_str(req)) and re.search(rf'\b{h}', self.get_req_as_str(req)):
                    found = True
                    break
            if found:
                ret += 1
        return ret

    def get_num_screenparameters(self, reqs):
        screen_resolution = ["screenheight", "screenwidth", "browserheight", "browserwidth",
                             "screendensity", "screen_res", "screen_param", "screenresolution", "browsertimeoffset"]
        ret = 0
        for req in reqs:
            for param in screen_resolution:
                ret += req.url.count(param)
        return ret

    def get_fp_func(self, blocked_reqs):
        keywords_fp = ["CanvasRenderingContext2D", "HTMLCanvasElement", "toDataURL",
                       "getImageData", "measureText", "font", "fillText", "strokeText",
                       "fillStyle", "strokeStyle", "HTMLCanvasElement.addEventListener",
                       "save", "restore"]

        total = 0
        total_eval_func = 0
        for req in blocked_reqs:
            try:
                resp = str(req.response.body)
            except:
                resp = ''

            for keyword in keywords_fp:
                total += resp.count(keyword)

            total_eval_func += resp.count('eval')
            total_eval_func += resp.count('function')

        return total, total_eval_func

    def build_network_features(self):
        self.fp_domain_list = utils.get_top_domain_list(self.page_delta)

        blocked_requests = self.page_delta.filter.get_session_blocked_reqs()
        self.features.num_requests_blocked = len(blocked_requests)
        num_requests_blocked_adjusted = self.features.num_requests_blocked
        if self.features.num_requests_blocked == 0:
            num_requests_blocked_adjusted = 1

        self.features.blocked_request_url_length = sum(
            [len(req.url) for req in blocked_requests]) / num_requests_blocked_adjusted
        self.features.percent_requests_blocked = len(
            blocked_requests) / len(self.page_delta.vanilla.requests)

        self.features.total_parameters = utils.get_total_parameters(
            blocked_requests)

        self.features.num_ad_dimensions = self.get_num_ad_dimension(
            blocked_requests)
        self.features.num_simicolon = sum(
            [req.url.count(';') for req in blocked_requests])
        self.features.num_screenparameters = self.get_num_screenparameters(
            blocked_requests)

        self.features.times_fp_in_blocked_requests = sum(
            [self.count_fp_domain(req) for req in blocked_requests])

        self.features.num_fp_req_blocked = sum([1 for req in blocked_requests if self.same_domain(
            extract(req.url).domain, self.fp_domain_list)])
        self.features.num_tp_req_blocked = self.features.num_requests_blocked - \
            self.features.num_fp_req_blocked

        self.features.num_ad_keywords = self.get_ad_keywords(blocked_requests)

        storage_values = self.extract_storage_values(
            self.page_delta.vanilla.rendered['cookies'], self.page_delta.vanilla.rendered['local_storage'], self.page_delta.vanilla.rendered['session_storage'])
        self.features.num_storage_values_out = self.get_storage_values_out(
            storage_values, blocked_requests)

        self.blocked_req_with_responses = self.find_blocked_requests(blocked_requests)[
            :len(blocked_requests)]

        self.features.fp_api_static, self.features.num_eval_func_keywords_static = self.get_fp_func(
            self.blocked_req_with_responses)
        self.features.total_response_size = 0
        for req in self.blocked_req_with_responses:
            try:
                self.features.total_response_size += len(req.response.body)
            except:
                pass
        self.features.avg_response_size = self.features.total_response_size / \
            num_requests_blocked_adjusted

        self.features.sensitive_fp_v, self.features.sensitive_tp_v = self.get_request_diff(
            self.page_delta.vanilla.requests, self.page_delta.blocked.requests)
        self.blocked_reqs = blocked_requests

        if debugging:
            code.interact('build_network_features',
                          local=dict(locals(), **globals()))

    def is_identical_req(self, r1, r2):
        return utils.similarity_score(r1.url, r2.url) > 0.9 and r1.method == r2.method

    def is_sensitive(self, req):

        parameters = parse_qs(urlparse(req.url).query)
        parameters = {p: val for p, val in parameters.items() if p not in [
            'v', 'version', 'ver']}

        contains_id = False
        for k in req.headers.keys():
            lower_key = k.lower()
            if lower_key == 'cookie' or lower_key == 'set-cookie':
                contains_id = True
                break
        if not contains_id:
            for p, val in parameters.items():
                if len(val) >= 5:
                    contains_id = True
                    break

        return req.method == 'POST' or len(parameters) or contains_id or len(req.body)

    def get_sensitive_requests(self, lst):
        sensitive = [[], []]
        for req in lst:
            is_sensitive = self.is_sensitive(req)
            is_fp = self.same_domain(
                extract(req.url).domain, self.fp_domain_list)

            if is_sensitive and is_fp:
                sensitive[0].append(req)
            elif is_sensitive and not is_fp:
                sensitive[1].append(req)

        return sensitive

    def same_domain(self, domain, domain_list):
        return domain in domain_list

    def get_request_diff(self, v, b):

        fp_v = [req for req in v if self.same_domain(
            extract(req.url).domain, self.fp_domain_list)]
        fp_b = [req for req in b if self.same_domain(
            extract(req.url).domain, self.fp_domain_list)]

        sensitive_fp_v, sensitive_tp_v = self.get_sensitive_requests(v)
        sensitive_fp_b, sensitive_tp_b = self.get_sensitive_requests(b)

        identical_i, identical_j = self.identical_pairs(
            sensitive_fp_v, sensitive_fp_b, self.is_identical_req)
        o1 = [v[idx]
              for idx in range(len(sensitive_fp_v)) if idx not in identical_i]
        o2 = [b[idx]
              for idx in range(len(sensitive_fp_b)) if idx not in identical_j]
        sensitive_fp_v = len(o1)

        identical_i, identical_j = self.identical_pairs(
            sensitive_tp_v, sensitive_tp_b, self.is_identical_req)
        o1 = [v[idx]
              for idx in range(len(sensitive_tp_v)) if idx not in identical_i]
        o2 = [b[idx]
              for idx in range(len(sensitive_tp_b)) if idx not in identical_j]
        sensitive_tp_v = len(o1)

        return sensitive_fp_v, sensitive_tp_v

    def build_performance_features(self):

        CDP_NETWORK_BINDING = 2
        CDP_NETWORK_CSS = 3

    def is_invalid_url(self, url, top_domain_list):
        b1 = any([sum(external_lists1.should_block(url, top_domain))
                 for top_domain in top_domain_list])
        b2 = any([sum(external_lists2.should_block(url, top_domain))
                 for top_domain in top_domain_list])
        return b1, b2

    def build_debug_features(self, blocked_reqs, top_domain_list):
        self.features.debug_name = os.path.basename(self.page_delta.save_path)
        self.features.debug_blocked_url = [req.url for req in blocked_reqs]
        self.features.debug_blocked_req_objs = codecs.encode(
            pickle.dumps(blocked_reqs), "base64").decode()

        ads_trackers = [self.is_invalid_url(
            url, top_domain_list) for url in self.features.debug_blocked_url]
        self.features.debug_is_ad = any([x for x, y in ads_trackers])
        self.features.debug_is_tracker = any([y for x, y in ads_trackers])

        self.features.debug_to_block = self.page_delta.filter.to_block
        self.features.debug_url = self.page_delta.url

    def build_features(self):

        self.parse_dom()
        self.build_appearnce_features()
        self.build_input_features()
        self.build_network_features()
        self.build_performance_features()
        self.build_tua_specific_features()
        self.build_features_on_responses()

        self.build_debug_features(self.blocked_reqs, self.fp_domain_list)

        features = [attr for attr in dir(self.features) if not callable(
            getattr(self.features, attr)) and not attr.startswith("__")]

        self.end = time.time()
        return self

    def save(self):
        utils.dump(self, os.path.join(
            self.page_delta.save_path, 'extended.pickle'))

    def dump_screenshots(self):
        output_dir = self.page_delta.save_path
        print("Dumping screenshots to %s" % output_dir)
        cv2.imwrite(os.path.join(output_dir, 'v.png'),
                    self.page_delta.vanilla.screenshot)
        cv2.imwrite(os.path.join(output_dir, 'b.png'),
                    self.page_delta.blocked.screenshot)


def load_existing(outpath):
    existing_csvs = [os.path.basename(fpath)[:-4]
                     for fpath in glob.glob('%s/*.csv' % (outpath))]
    if not existing_csvs:
        code.interact('No existing csvs', local=dict(locals(), **globals()))
        return

    return existing_csvs


async def async_load_single_pagedelta(pagedelta_fname, idx):
    timeout_time = 60 * 15
    with trio.move_on_after(timeout_time) as scope:
        ret = await load_single_pagedelta((pagedelta_fname, idx))
        return ret

    if scope.cancelled_caught:
        print("Timeout after %ds: %s" % (timeout_time, pagedelta_fname))

    return


async def load_single_pagedelta(args):
    if isinstance(args, tuple):
        pagedelta_fname, idx = args
    elif isinstance(args, str):
        pagedelta_fname = args
        idx = -1

    if idx != -1:

        pass
    else:

        pass

    try:
        g = utils.load(pagedelta_fname)
        extended_pagedelta = ExtendedPageDelta(g)
        row = []
        if extended_pagedelta.page_delta.vanilla is None and extended_pagedelta.page_delta.vanilla_retry:
            retry_count = extended_pagedelta.page_delta.retry_count if 'retry_count' in dir(
                extended_pagedelta.page_delta) else len(extended_pagedelta.page_delta.vanilla_retry)
            for i in range(retry_count):
                for j in range(retry_count):
                    if debugging:
                        if (i, j) != (debug_i, debug_j):
                            continue
                        else:
                            print("Debugging retry idx %d %d" %
                                  (debug_i, debug_j))
                            code.interact('debugging load_single_pagedelta', local=dict(
                                locals(), **globals()))

                    extended_pagedelta.page_delta.filter.check_sessions()

                    if not extended_pagedelta.page_delta.vanilla_retry[i].is_valid_visit():

                        continue
                    extended_pagedelta.page_delta.vanilla = extended_pagedelta.page_delta.vanilla_retry[
                        i]

                    if not extended_pagedelta.page_delta.filter.is_valid_block(j) or not extended_pagedelta.page_delta.blocked_rety[j].is_valid_visit():

                        continue
                    extended_pagedelta.page_delta.blocked = extended_pagedelta.page_delta.blocked_rety[
                        j]

                    extended_pagedelta.features.debug_vanilla_retry = i
                    extended_pagedelta.features.debug_blocked_retry = j
                    extended_pagedelta = extended_pagedelta.build_features()

                    row.append(copy.deepcopy(
                        vars(extended_pagedelta.features)))
        else:
            extended_pagedelta = extended_pagedelta.build_features()
            extended_pagedelta.features.debug_vanilla_retry = -1
            extended_pagedelta.features.debug_blocked_retry = -1
            row.append(vars(extended_pagedelta.features))

        row = pd.DataFrame(row)

        if tmp_file_path:
            row.to_csv('%s/%s.csv' % (tmp_file_path, os.path.basename(
                os.path.dirname(pagedelta_fname))), sep=',', index=False)
        else:
            code.interact('debuging single pagedetla',
                          local=dict(locals(), **globals()))
            print('csv not written to disk, assuming debugging')

    except KeyboardInterrupt:
        print("KeyboardInterrupt %s " % pagedelta_fname)
        raise

    except:
        traceback.print_exc()
        print("Error with %s" % pagedelta_fname)


def load_pagedeltas(pagedelta_fnames):
    random.shuffle(pagedelta_fnames)
    if debugging:
        print('Debug  ', debug_fname)
        pagedelta_fnames = list(
            filter(lambda x: debug_fname in x, pagedelta_fnames))

    num_instances = 1

    if debugging:
        num_instances = 1

    assert (tmp_file_path), 'tmp_file_path is None'
    if os.path.isdir(tmp_file_path):

        pass
    else:
        os.mkdir(tmp_file_path)

    print('Loading using %d processes on %d pds saving to %s' %
          (num_instances, len(pagedelta_fnames), tmp_file_path))
    if num_instances == 1:
        for pd_fnames in pagedelta_fnames:
            ret = trio.run(load_single_pagedelta, pd_fnames)
    else:
        call_method = 1

        with Pool(processes=num_instances) as pool:
            if call_method == 1:
                funcs = [async_load_single_pagedelta] * len(pagedelta_fnames)
                args = zip(funcs, pagedelta_fnames,
                           range(len(pagedelta_fnames)))
                total = pool.starmap(trio.run, args)

            elif call_method == 2:
                idxs = list(range(len(pagedelta_fnames)))
                result = pool.map(load_single_pagedelta,
                                  zip(pagedelta_fnames, idxs))

            elif call_method == 4:
                idxs = list(range(len(pagedelta_fnames)))
                list(pool.imap_unordered(
                    load_single_pagedelta, zip(pagedelta_fnames, idxs)))

            elif call_method == 3:
                running_procs = [None] * num_instances

                for pd_idx, pd in enumerate(pagedelta_fnames):

                    while all([not (proc_status is None or proc_status.poll() is None) for proc_status in running_procs]):
                        time.sleep(30)

                    next_idx = -1
                    for idx, proc_status in enumerate(running_procs):
                        if proc_status is None or proc_status.returncode is not None:
                            next_idx = idx
                            break

                    assert (next_idx != -1), 'must have an open slot'
                    running_procs[next_idx] = subprocess.Popen(
                        ['python3', 'page_delta.py', 'load_single', str(pd), str(pd_idx)], shell=True)

                for proc_status in running_procs:
                    if proc_status is None:
                        continue
                    proc_status.wait()


def find_last_fname(path):
    fname = data_fname(path)
    existings = glob.glob(
        '%s/%s*' % (os.path.dirname(fname), os.path.basename(fname).split('-')[0]))
    assert (existings), 'no file found for %s' % fname

    existings.sort(key=os.path.getctime, reverse=True)
    last = existings[0]
    return last


def build_datapoints(path='./page_slimmer/easy_privacy_0-1k.parsed', _load_existing=True, load_last=False, first_k=-1):
    if path.endswith('/'):
        path = path[:-1]
    first_k = int(first_k)

    global tmp_file_path
    if first_k == -1:
        tmp_file_path = os.path.join(os.path.dirname(path).replace(
            'disk7', 'disk6'), os.path.basename(path) + '_tmp_results')
    else:
        tmp_file_path = os.path.join(os.path.dirname(
            path), os.path.basename(path) + '_tmp_results_%d' % first_k)

    if load_last:
        last_data = find_last_fname(path)
        print("Combining %s" % last_data)
        total = pd.read_csv(last_data, sep=',')
        total = utils.combine_retries(total)
        write_to_data_folder(total, path)
        return

    if isinstance(_load_existing, str):
        _load_existing = int(_load_existing)

    print('working on [continue_mode:%d] %s, res to %s' %
          (int(_load_existing), path, tmp_file_path))
    existing = None
    if _load_existing:
        existing = load_existing(tmp_file_path)

    total = []
    pagedelta_fnames = utils.find_all_pagedeltas(
        path, existing=existing, first_k=first_k)

    skip_first = 0
    if skip_first:
        pagedelta_fnames = pagedelta_fnames[skip_first:]

    pagedelta_fnames.sort()
    pagedelta_fnames = pagedelta_fnames

    total = load_pagedeltas(pagedelta_fnames)

    if total is None:
        total = collect_files(tmp_file_path)

    total = pd.concat([existing, total])

    total = utils.combine_retries(total)

    return True


def collect_files(dir_path):
    total = []
    pds = os.listdir(dir_path)

    for fname in pds:
        try:
            rows = pd.read_csv(os.path.join(dir_path, fname), sep=',')
            total.append(rows)
        except:
            traceback.print_exc()

    total = pd.concat(total)

    print("Collected %d rows from %d files at %s" %
          (total.shape[0], len(pds), dir_path))
    return total


def data_fname(path):
    return 'data/%s-%s.csv' % (os.path.basename(path), datetime.date.today().strftime('%Y%m%d'))


def write_to_data_folder(df, path):
    df.to_csv(data_fname(path), sep=',', index=False)




def req_path(row):
    if isinstance(row, str):
        o = urlparse(row)
    else:
        o = urlparse(row['url'])
    url_without_query_string = o.scheme + "://" + o.netloc + o.path
    return url_without_query_string


if __name__ == '__main__':
	build_datapoints(sys.argv[1], int(sys.argv[2]),
						int(sys.argv[3]), sys.argv[4])
