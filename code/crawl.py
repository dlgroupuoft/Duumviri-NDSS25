from keras_preprocessing.sequence import pad_sequences
from collections import Counter
import csv
import pandas as pd
import sys
from pagedelta import PageDelta
import pagedelta
import logging
import requests
import shutil
from urllib.parse import urlparse, parse_qs

import traceback
import random
from bs4 import BeautifulSoup
from tldextract import extract
from Cache import Cache
import enchant

import re
from selenium.webdriver.common.by import By
import trio
import numpy as np
import cv2
import ast
from urllib.parse import urlparse, parse_qs
import seleniumwire
from datetime import timedelta, datetime
import selenium
import os
import copy
import utils
from difflib import SequenceMatcher
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import code
import glob
import time
from xml.sax import default_parser_list
from wsgiref import validate
from functools import total_ordering
from collections.abc import Iterable
from multiprocessing import Pool, Array
import pytz
from seleniumwire.request import Response


def find_highest_idx(to_check_path, pattern='*_2.png'):
	high = 0
	existing_idxs = set()
	for path in glob.glob(os.path.join(to_check_path, pattern)):
		curr = int(os.path.basename(path).split('_')[0])
		existing_idxs.add(curr)
		if curr > high:
			high = curr
	return high, list(existing_idxs)


class DataCollection():
	def __init__(self):

		self.prefilter_histories = {}
		self.english_dict = enchant.Dict("en_US")
		self.wire_options = {'disable_encoding': True,
			'ignore_http_methods': ['OPTIONS'], 'verify_ssl': False}
		self.ad_dimension_pattern = re.compile(r"\d{2,4}[^\d]\d{2,4}")
		self.archive_usage_path = 'archive_usage.txt'
		self.url_regex = re.compile(r"""(?P<url>\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@))))""")
		self.empty_response = Response(
			status_code=0, reason="", headers={}, body="")
		self.external_lists = None
		self.before_unload_script = """
		function before_unload1() {
			var secs = new Date().getTime() / 1000
			before_unload(secs.toString());
		}
		window.addEventListener('beforeunload', before_unload1);
		"""
		self.focus_js = """
		var iframes = document.querySelectorAll('iframe')
		for (let i = 0; i < iframes.length; i++) {
			iframe = iframes[i];
			iframe.contentWindow.focus();
			iframe.scrollIntoView();
		}
		window.scrollTo(0, 0);
		"""

	def quit_driver(self, driver):
		if 'display' in dir(driver):
			driver.display.stop()
		driver.quit()


	def init_path(self, dir):
		self.save_path = dir
		os.makedirs(self.save_path, exist_ok=True)

	def is_chrome_scripts(self, url):
		return url.startswith('chrome-extension://') or url.startswith('chrome://')

	def get_full_scripts(self, driver, top_domain_list):

		fp_scripts = set([s.url for s in driver.script_mapping if urlparse(
			s.url).path.endswith('.js') and not self.is_chrome_scripts(s.url)])
		ret = set([url for url, _, _, _ in self.get_candidates_for_masking(driver)])
		return list(fp_scripts.union(ret))

	def get_parsed_scripts(self, driver):

		scripts = set([s.url for s in driver.script_mapping if urlparse(
			s.url).path.endswith('.js') and not self.is_chrome_scripts(s.url)])
		tracking_messages = set(
			[url for url, _, _, _ in self.get_candidates_for_masking(driver)])
		return list(scripts.union(tracking_messages))

	def get_tp_scripts(self, driver, top_domain_list):

		return [s.url for s in driver.script_mapping if urlparse(s.url).path.endswith('.js') and '.'.join(extract(s.url)[1:]) not in top_domain_list and not self.is_chrome_scripts(s.url)]


	def get_candidates_for_initiating_script(self, driver, initiating_script_url, pd):
		ret = []

		for event in pd.vanilla_retry[0].rendered['cdp_network'][0]:
			if event.initiator.type_ == 'script':
				init_urls = [
					callframe.url for callframe in event.initiator.stack.call_frames]
				if initiating_script_url in init_urls:

					ret += [event.request.url]

		return ret

	def get_candidates_for_masking(self, driver, pd=None, filter_out_e1=False, url=""):
		ret = []

		top_domain_list = list(
			set(['.'.join(extract(url)[1:]), '.'.join(extract(driver.current_url)[1:])]))

		if filter_out_e1:
			req_resps = [(req, req.response)
						  for req in driver.requests if not self.is_invalid_url(req.url, top_domain_list)]
		else:
			req_resps = [(req, req.response) for req in driver.requests]

		storage_values = pagedelta.ExtendedPageDelta(
			None).extract_storage_values(*utils.get_storage_values(driver))
		deduplication = set()
		no_analyze, num_same_resp = 0, 0
		self.tried_urls = set()
		cookie_fields = set()
		tried_cookie_name = set()

		for req, response in req_resps:
			req_url = req.url

			if not req_url.startswith('http'):
				continue
			parameters = parse_qs(urlparse(req_url).query)
			cookie_header = req.headers.get(
				'cookie', '') or req.headers.get('Cookie', '')
			include_cookie = False
			if cookie_header and include_cookie:
				cookies_parsed = utils.parse_raw_cookie(cookie_header)
				cookies_parsed = {k: [v] for k, v in cookies_parsed.items()}
				parameters.update(cookies_parsed)
				for k in cookies_parsed:
					cookie_fields.add(k)
			for p_name, p_values in parameters.items():
				for p_value in p_values:
					key_value = req_url+p_name

					if key_value not in deduplication and (utils.param_from_storage(p_value, storage_values) or utils.param_is_id(p_value)) and not self.english_dict.check(p_value.strip(";,/?:@&=+$-_.!~*'()#")) and not p_name in tried_cookie_name:
						deduplication.add(key_value)
						self.tried_urls.add(req_url)
						same_resp=-1
						if p_name in cookie_fields:
							req_url='*'

						ret.append((req_url, p_name, p_value, same_resp))

						if p_name in cookie_fields:
							tried_cookie_name.add(p_name)

					else:
						no_analyze += 1
		if pd:
			pd.calculate_robot_info_statistics(self.tried_urls)
		ret=list(set(ret))
		return ret

	
	def generic_collection_routine(self, urls, idxs, id, additionals):
		try:
			time.sleep(int(id))
			assert (isinstance(id, int) and len(additionals) ==
					4), "Id: %s, additionals: %s" % (str(id), str(additionals))

			total, status_file_path, iterate_control, try_max=additionals
			driver1=None

			orig_urls=urls
			if isinstance(urls, pd.DataFrame):
				to_blocks=urls.iloc[:, 1:].values.tolist()
				urls=list(urls.iloc[:, 0])

			urllist=list(enumerate(zip(idxs, urls)))
			random.shuffle(urllist)
			for url_idx, (idx, url) in urllist:
				try:
					url=self.add_protocol(url)
					print("Iterate_control:%s with trymax:%s %d/%d %s" %
						  (iterate_control, str(try_max), idx, total, url))
					s=time.time()
					same_page_idx=0
					page_delta=PageDelta(os.path.join(self.save_path, '%d_%s_%d' % (
						idx, utils.str_to_fname(url), same_page_idx)), url, idx, "")
					cache=Cache()
					driver1=trio.run(self.get_url_with_scripts_mapping, driver1, url, cache)
					driver1=page_delta.save_vanilla(
						driver1, cache, retry_func=self.get_url_with_scripts_mapping)

					page_delta_with_vanilla=copy.deepcopy(page_delta)
					cache_with_vanilla=copy.deepcopy(cache)
					top_domain_list=list(
						set(['.'.join(extract(url)[1:]), '.'.join(extract(driver1.current_url)[1:])]))
					if iterate_control == 'full':
						fp_scripts=self.get_full_scripts(driver1, top_domain_list)
					elif iterate_control == 'random_crawl':
						fp_scripts=self.get_parsed_scripts(driver1)
					elif iterate_control == 'tp_only':
						fp_scripts=self.get_tp_scripts(driver1, top_domain_list)
					elif iterate_control == 'parameter_masking':
						fp_scripts=self.get_candidates_for_masking(
							driver1, page_delta, filter_out_e1=1, url=url)
					elif iterate_control == 'specified':
						fp_scripts=[to_blocks[url_idx]]
					elif iterate_control == 'initiating_script':
						fp_scripts=self.get_candidates_for_initiating_script(
							driver1, to_blocks[url_idx][0], page_delta)
					else:
						print('Unknown iterate control %s' % iterate_control)
						sys.exit(1)

					tried=0
					original_len=len(fp_scripts)
					while True:
						cache=cache_with_vanilla
						if not fp_scripts or tried == try_max:

							break

						try:
							to_block=fp_scripts.pop()

							tried += 1

							cache.set_block_url(to_block, top_domain_list)

							print("[%d/%d] [%d/%d] Working on %s %s" % (idx, total,
								  original_len - len(fp_scripts), original_len, url, to_block))

							driver1=trio.run(self.get_url_with_scripts_mapping, driver1, url, cache)
							if not (len(cache.get_last_blocked_requests())):
								print("[%d/%d] [%d/%d] No hit %s %s" % (idx, total,
									  original_len - len(fp_scripts), original_len, url, to_block))
								continue

							if same_page_idx != 0:
								page_delta=copy.deepcopy(page_delta_with_vanilla)
								page_delta.save_path=os.path.join(self.save_path, '%d_%s_%d' % (
									idx, utils.str_to_fname(url), same_page_idx))
							driver1=page_delta.save_blocked(
								driver1, cache, retry_func=self.get_url_with_scripts_mapping)
							page_delta.save(driver1, tried, original_len)
							same_page_idx += 1
						except KeyboardInterrupt:
							raise
						except AssertionError as msg:
							print(msg)
						except:
							traceback.print_exc()
				except KeyboardInterrupt:
					self.quit_driver(driver1)
					raise
				except:
					traceback.print_exc()
		except:
			traceback.print_exc()
			print("Crawler crashed [%d]" % id)
   
	def collect_parameter_one(self, url):
		urls=[url]
		dir='./parameter_pagedelta'
		self.init_path(dir)
		highest_idx, existing_idxs=find_highest_idx(self.save_path, pattern='*')

		self.split_data_and_run(urls, existing_idxs, self.generic_collection_routine,
								num_instances=1, additionals=['parameter_masking', np.inf])
	
	def collect_random_crawl(self, dir='/vail/tranco_5k_06_2024', urls=[]):
		self.init_path(dir)
		if not urls:
			urls=pd.read_csv('./example_pages/tranco-1m.csv', header=None)[1].tolist()

			urls=urls[:5000]
		highest_idx, existing_idxs=find_highest_idx(self.save_path, pattern='*')
		self.split_data_and_run(urls, existing_idxs, self.generic_collection_routine,
								num_instances=30, additionals=['random_crawl', 150])

	def split_data_and_run(self, our_dataset, continue_or_fill, to_para_func, num_instances=6, additionals=[]):
		"""
		continue_or_fill represents two modes
				- continue_or_fill is a interger when in continue mode, the algo treats the integer as the starting idx
				- continue_or_fill is a list of exisitng integers when in fill mode, the algo ignores any idx in the list and fills the blank
		"""
		if isinstance(continue_or_fill, int):
			print("Continue mode from %d of %d" % (continue_or_fill, len(our_dataset)))
			start_i=max(continue_or_fill, 0)

			code.interact('continue', local=dict(locals(), **globals()))
			input_vals=our_dataset[start_i:] if isinstance(
				our_dataset, list) else our_dataset.iloc[start_i:, :]
			idxs=list(range(start_i, start_i + len(input_vals)))
		else:
			assert (isinstance(continue_or_fill, list)
				   ), 'split_data_and_run - Unknown continue_or_fill %s ' % (str(continue_or_fill)[:10])
			skip_first=0

			print("Filling mode, skipping %d of %d, and skip first %d" %
				  (len(continue_or_fill), len(our_dataset), skip_first))

			if isinstance(our_dataset, list):

				idx_lst=[(i, val) for i, val in enumerate(
					our_dataset) if i not in continue_or_fill]
				if not idx_lst:
					return
				idxs, input_vals=zip(*idx_lst)
			else:
				assert (isinstance(our_dataset, pd.DataFrame))
				mask=~our_dataset.index.isin(continue_or_fill)
				input_vals=our_dataset[mask]
				idxs=input_vals.index.tolist()
		assert (len(input_vals) == len(idxs)), 'input_vals:%d idxs:%d' % (
			len(input_vals), len(idxs))
		input_vals=utils.equal_sized_chunks(input_vals, num_instances)
		idxs=utils.equal_sized_chunks(idxs, num_instances)
		assert (len(input_vals) == num_instances and len(idxs) == num_instances)
		for i in range(num_instances):
			assert (len(input_vals[i]) == len(idxs[i]))

		total=len(our_dataset)
		additionals.insert(0, total)
		status_file_path='./git_related/status_files/%s' % to_para_func.__name__
		additionals.insert(1, status_file_path)
		ids=list(range(num_instances))

		additionals=[additionals] * num_instances
		if num_instances == 1:
			to_para_func(input_vals[0], idxs[0], ids[0], additionals[0])
		else:
			with Pool(num_instances) as p:
				p.starmap(to_para_func, list(
					zip(input_vals, idxs, ids, additionals)))

	def add_protocol(self, url):
		url=url.strip()
		if not url.startswith('http'):
			return 'https://%s' % url
		return url



	async def handle_script_parsed(self, receive_channel):
		async with receive_channel:
			async for value in receive_channel:
				self.script_mapping.append(value)

	async def handle_request_sent(self, receive_channel):
		async with receive_channel:
			async for value in receive_channel:
				self.sent_requests.append(value)
	async def handle_request_sent2(self, receive_channel):
		async with receive_channel:
			async for value in receive_channel:
				self.sent_requests2.append(value)

	async def handle_response(self, receive_channel):
		async with receive_channel:
			async for value in receive_channel:
				self.received_responses.append(value)
	async def handle_response2(self, receive_channel):
		async with receive_channel:
			async for value in receive_channel:
				self.received_responses2.append(value)
	async def handle_binding(self, receive_channel):
		async with receive_channel:
			async for value in receive_channel:
				self.binding_called.append(value)
	async def handle_css_added(self, receive_channel):
		async with receive_channel:
			async for value in receive_channel:
				self.css_added.append(value)
	async def handle_frame_attached(self, receive_channel):
		async with receive_channel:
			async for value in receive_channel:
				self.frame_attached.append(value)

	async def handle_log(self, receive_channel):
		async with receive_channel:
			async for value in receive_channel:
				self.log_added.append(value)

	async def handle_console_api(self, receive_channel):
		async with receive_channel:
			async for value in receive_channel:
				self.console_api.append(value)

	async def handle_frame_navigated(self, receive_channel):
		async with receive_channel:
			async for value in receive_channel:
				print("frame navigated %s" % str(value))

	async def handle_loading_finished(self, receive_channel):
		async with receive_channel:
			async for value in receive_channel:

				self.request_loading_finished.append(value)
	async def __aexit__(self):
		pass

	def reset_render_time_features(self, driver):
		if 'requests' in dir(driver):
			del driver.requests
		if 'script_mapping' in dir(driver):
			del driver.script_mapping
		if 'cdp_network' in dir(driver):
			del driver.cdp_network
		if 'get_url' in dir(driver):
			del driver.get_url
		if 'scroll_height' in dir(driver):
			del driver.scroll_height

	def reset_browser(self, driver):

		driver.delete_all_cookies()
		if not (driver.get_cookies() == []):
			code.interact('cookies not deleted', local=dict(locals(), **globals()))

		ret=driver.execute_cdp_cmd('Storage.clearCookies', {})
		assert (driver.execute_cdp_cmd('Storage.getCookies', {})['cookies'] == [])
		if 'script_mapping' in dir(driver) and 'data:text/html;charset=utf-8,' not in driver.current_url:
			try:
				utils.execute_script_with_catch2(driver, 'window.localStorage.clear(); ')
				utils.execute_script_with_catch2(driver, 'window.sessionStorage.clear(); ')
			except:
				traceback.print_exc()

		if 'request_interceptor' in dir(driver) and driver.request_interceptor is not None:
			del driver.request_interceptor
		self.reset_render_time_features(driver)

		self.script_mapping=[]
		self.sent_requests=[]
		self.sent_requests2=[]
		self.received_responses=[]
		self.received_responses2=[]
		self.binding_called=[]
		self.css_added=[]
		self.frame_attached=[]
		self.log_added=[]
		self.request_loading_finished=[]
		self.console_api=[]

	async def execute_cdp(self, session, command):
		ret=await session.execute(command)

		return ret

	async def get_url_with_scripts_mapping(self, driver1, url, cache=None, proxy_on=False, timeout=60, ):

		if driver1 is not None:
			driver1.quit()

		driver1=utils.init_chrome_with_cdp(
			extension_path='./page_slimmer/adhighlighter/Perceptual_Ad_Highlighter', no_proxy=not proxy_on)
		assert (driver1)
		target_idx=0
		self.driver=driver1
		async with driver1.bidi_connection() as session:
			session, devtools=session.session, session.devtools

			targets=await self.execute_cdp(session, devtools.target.get_targets())
			main_target=[target for target in targets if target.type_ == 'page'][0]
			target_idx=targets.index(main_target)

		self.reset_browser(driver1)

		if cache:
			cache.begin(driver1)

		timeout_time=timeout

		with trio.move_on_after(timeout_time) as cancel_scope:
			async with driver1.bidi_connection(target_idx=target_idx) as session:

				session, devtools=session.session, session.devtools

				await self.execute_cdp(session, devtools.emulation.set_focus_emulation_enabled(True))
				await self.execute_cdp(session, devtools.dom.enable())

				script_parsed_listener=session.listen(devtools.debugger.ScriptParsed)
				request_sent_listener=session.listen(devtools.network.RequestWillBeSent)
				request_sent_listener2=session.listen(
					devtools.network.RequestWillBeSentExtraInfo)
				response_listener=session.listen(devtools.network.ResponseReceived)
				response_listener2=session.listen(
					devtools.network.RequestWillBeSentExtraInfo)

				binding_listener=session.listen(devtools.runtime.BindingCalled)
				css_added_listener=session.listen(devtools.css.StyleSheetAdded)
				frame_attached_listener=session.listen(devtools.page.FrameAttached)
				log_listener=session.listen(devtools.log.EntryAdded)
				console_api_listener=session.listen(devtools.runtime.ConsoleAPICalled)
				LoadingFinished_listener=session.listen(devtools.network.LoadingFinished)
				async with trio.open_nursery() as nursey:
					nursey.start_soon(self.handle_script_parsed, script_parsed_listener)
					nursey.start_soon(self.handle_request_sent, request_sent_listener)
					nursey.start_soon(self.handle_request_sent2, request_sent_listener2)
					nursey.start_soon(self.handle_response, response_listener)
					nursey.start_soon(self.handle_response2, response_listener2)
					nursey.start_soon(self.handle_binding, binding_listener)
					nursey.start_soon(self.handle_css_added, css_added_listener)
					nursey.start_soon(self.handle_frame_attached, frame_attached_listener)
					nursey.start_soon(self.handle_log, log_listener)
					nursey.start_soon(self.handle_console_api, console_api_listener)
					nursey.start_soon(self.handle_loading_finished, LoadingFinished_listener)
					await self.execute_cdp(session, devtools.dom.enable())
					await self.execute_cdp(session, devtools.network.clear_browser_cache())
					await self.execute_cdp(session, devtools.log.enable())
					await self.execute_cdp(session, devtools.debugger.enable())
					await self.execute_cdp(session, devtools.runtime.enable())
					await self.execute_cdp(session, devtools.page.enable())
					await self.execute_cdp(session, devtools.css.enable())
					await self.execute_cdp(session, devtools.network.enable())
					await self.execute_cdp(session, devtools.network.clear_browser_cache())

					await self.execute_cdp(session, devtools.runtime.add_binding('before_unload'))
					if url:
						begin=time.time()
						ret=driver1.get(url)

						driver1.load_time=time.time() - begin
						window_id, bounds=await session.execute(devtools.browser.get_window_for_target())
						bounds.width, bounds.height, bounds.windowState=1920, 1080, 'normal'
						await session.execute(devtools.browser.set_window_bounds(window_id, bounds))

						await self.execute_cdp(session, devtools.runtime.evaluate(self.before_unload_script))
						focus=True
						if focus:
							await self.execute_cdp(session, devtools.runtime.evaluate(self.focus_js))
						else:
							print('no focus')

						driver1.get_url=url
					else:
						driver1.find_element(By.TAG_NAME, 'body').send_keys(
							Keys.CONTROL + Keys.F5)

					await trio.sleep(30)
					cancel_scope.cancel()

		if not ('get_url' in dir(driver1)):
			print('get_url not in dir(driver1), this means the page timedout and some CDP commands are not executed')

		page_source_thr=500 if proxy_on else 1500
		if len(driver1.page_source) < page_source_thr:
			try:
				reason=BeautifulSoup(driver1.page_source, 'html.parser').find('title').text
			except:
				reason=BeautifulSoup(driver1.page_source, 'html.parser').text[:100]
			assert (False), 'Failed to load due to %s' % reason
		self.stop_blocking(driver1, cache)
		driver1.script_mapping=self.script_mapping
		driver1.cdp_network=(self.sent_requests, self.received_responses, self.binding_called, self.css_added, self.frame_attached,
							 self.log_added, self.console_api, self.sent_requests2, self.received_responses2, self.request_loading_finished)

		return driver1

	def stop_blocking(self, driver, cache):

		if cache:
			cache.end(driver)

		if 'request_interceptor' in dir(driver) or 'response_interceptor' in dir(driver):
			del driver.request_interceptor
			del driver.response_interceptor
   
	def is_invalid_url(self, url, top_domain_list):

		if self.external_lists is None:
			self.external_lists=utils.ExternalFilters([1, 1, 0, 0, 0, 0])
		if not url:
			return False

		return any([any(self.external_lists.should_block(url, domain)) for domain in top_domain_list])

	def retry_func(self, driver, url, cache=None):
		trio.run(self.get_url_with_scripts_mapping, driver, url, cache)
	
 
if __name__ == '__main__':
	if sys.argv[1] == '1':
		DataCollection().collect_random_crawl('./pagedelta/', [sys.argv[2]])
	elif sys.argv[1] == '2':
		DataCollection().collect_parameter_one(sys.argv[2])
