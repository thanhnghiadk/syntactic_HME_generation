################################################################
# inkml.py - InkML parsing lib
#
# Original Author: H. Mouchere, Feb. 2014
# Updated Author: Sivic, Feb. 2020
# Copyright (c) 2014, Harold Mouchere
################################################################

import xml.etree.ElementTree as ET
from transformation import transformation
import numpy as np
import math

class Segment(object):
    """Class to reprsent a Segment compound of strokes (id) with an id and label."""
    __slots__ = ('id', 'label', 'strId')

    def __init__(self, *args):
        if len(args) == 3:
            self.id = args[0]
            self.label = args[1]
            self.strId = args[2]
        else:
            self.id = "none"
            self.label = ""
            self.strId = set([])


class Inkml(object):
    """Class to represent an INKML file with strokes, segmentation and labels"""
    __slots__ = ('fileName', 'strokes', 'strkOrder', 'segments', 'truth', 'UI', "strokes_num", "x_mean", "y_mean", "x_min", "y_max", "width", "height");

    NS = {'ns': 'http://www.w3.org/2003/InkML', 'xml': 'http://www.w3.org/XML/1998/namespace'}

    ##################################
    # Constructors (in __init__)
    ##################################
    def __init__(self, *args):
        """can be read from an inkml file (first arg)"""
        self.fileName = None
        self.strokes = {}
        self.strkOrder = []
        self.strokes_num = {}
        self.x_mean = 0
        self.y_mean = 0
        self.x_min = 0
        self.y_max = 0
        self.width = 0
        self.height = 0
        self.segments = {}
        self.truth = ""
        self.UI = ""
        if len(args) == 1:
            self.fileName = args[0]
            self.loadFromFile()

    def fixNS(self, ns, att):
        """Build the right tag or element name with namespace"""
        return '{' + Inkml.NS[ns] + '}' + att

    def loadFromFile(self):
        """load the ink from an inkml file (strokes, segments, labels)"""
        tree = ET.parse(self.fileName)
        # # ET.register_namespace();
        root = tree.getroot()
        for info in root.findall('ns:annotation', namespaces=Inkml.NS):
            if 'type' in info.attrib:
                if info.attrib['type'] == 'truth':
                    self.truth = info.text.strip()
                if info.attrib['type'] == 'UI':
                    self.UI = info.text.strip()
        for strk in root.findall('ns:trace', namespaces=Inkml.NS):
            self.strokes[strk.attrib['id']] = strk.text.strip()
            self.strkOrder.append(strk.attrib['id'])
        segments = root.find('ns:traceGroup', namespaces=Inkml.NS)
        if segments is None or len(segments) == 0:
            print("No segmentation info")
            return
        for seg in (segments.iterfind('ns:traceGroup', namespaces=Inkml.NS)):
            id = seg.attrib[self.fixNS('xml', 'id')]
            label = seg.find('ns:annotation', namespaces=Inkml.NS).text
            strkList = set([])
            for t in seg.findall('ns:traceView', namespaces=Inkml.NS):
                strkList.add(t.attrib['traceDataRef'])
            self.segments[id] = Segment(id, label, strkList)

    def getInkML(self, file):
        """write the ink to an inkml file (strokes, segments, labels)"""
        outputfile = open(file, 'w')
        outputfile.write(
            "<ink xmlns=\"http://www.w3.org/2003/InkML\">\n<traceFormat>\n<channel name=\"X\" type=\"decimal\"/>\n<channel name=\"Y\" type=\"decimal\"/>\n</traceFormat>")
        outputfile.write("<annotation type=\"truth\">" + self.truth + "</annotation>\n")
        outputfile.write("<annotation type=\"UI\">" + self.UI + "</annotation>\n")
        for id in sorted(self.strokes.keys(), key=lambda x: float(x)):
            outputfile.write("<trace id=\"" + id + "\">\n" + self.strokes[id] + "\n</trace>\n")
        outputfile.write("<traceGroup>\n")
        for (id, s) in self.segments.items():
            outputfile.write("\t<traceGroup xml:id=\"" + id + "\">\n")
            outputfile.write("\t\t<annotation type=\"truth\">" + s.label + "</annotation>\n")
            for t in s.strId:
                outputfile.write("\t\t<traceView traceDataRef=\"" + t + "\"/>\n")
            outputfile.write("\t</traceGroup>\n")
        outputfile.write("</traceGroup>\n</ink>")
        outputfile.close()

    def isRightSeg(self, seg):
        """return true is the set seg is an existing segmentation"""
        for s in self.segments.values():
            if s.strId == seg:
                return True
        return False

    def getInkMLwithoutGT(self, withseg, file):
        """write the ink to an inkml file (strokes, segments, labels)"""
        outputfile = open(file, 'w')
        outputfile.write(
            "<ink xmlns=\"http://www.w3.org/2003/InkML\">\n<traceFormat>\n<channel name=\"X\" type=\"decimal\"/>\n<channel name=\"Y\" type=\"decimal\"/>\n</traceFormat>")
        outputfile.write("<annotation type=\"UI\">" + self.UI + "</annotation>\n")
        for id in sorted(self.strokes.keys(), key=lambda x: float(x)):
            outputfile.write("<trace id=\"" + id + "\">\n" + self.strokes[id] + "\n</trace>\n")
        if withseg:
            outputfile.write("<traceGroup>\n")
            for id in sorted(self.segments.keys(), key=lambda x: float(x) if x.isdigit() else x):
                outputfile.write("\t<traceGroup xml:id=\"" + id + "\">\n")
                outputfile.write("\t\t<annotation type=\"truth\">" + self.segments[id].label + "</annotation>\n")
                for t in sorted(self.segments[id].strId, key=lambda x: float(x) if x.isdigit() else x):
                    outputfile.write("\t\t<traceView traceDataRef=\"" + t + "\"/>\n")
                outputfile.write("\t</traceGroup>\n")
            outputfile.write("</traceGroup>")
        outputfile.write("</ink>")
        outputfile.close()

    # filter storkes by selected ids
    def filter_strokes(self, ids, new_truth):
        # assign new truth
        self.truth = new_truth

        # remove unused ids
        _strokes = {}
        for id in sorted(self.strokes.keys(), key=lambda x: float(x)):
            if id in ids:
                # outputfile.write("<trace id=\""+id+"\">\n"+self.strokes[id]+"\n</trace>\n")
                _strokes[id] = self.strokes[id]

        self.strokes = _strokes

        # remove unused segment
        _new_segments = {}
        for _seg_name in self.segments:
            _seg = self.segments[_seg_name]
            _str_id = list(_seg.strId)[0]
            if _str_id in ids and _seg not in _new_segments:
                _new_segments[_seg_name] = _seg
        self.segments = _new_segments

        pass

    # stroke ids data from string to number
    def parse_stroke_data(self, selected_ids):
        self.strokes_num = {}

        _x_pts = []
        _y_pts = []
        for _id in self.strokes.keys():
            if _id in selected_ids:
                _x_points = []
                _y_points = []
                # get real id
                _s_data = self.strokes[_id]
                _s_cords = _s_data.split(",")
                for _cord in _s_cords:
                    _term = _cord.split()
                    if "." in _s_data:
                        _x_points.append(int(float(_term[1])*100000))
                        _y_points.append(int(float(_term[0])*100000))
                    else:
                        _x_points.append(int(_term[1]))
                        _y_points.append(int(_term[0]))
                _x_pts = _x_pts + _x_points
                _y_pts = _y_pts + _y_points
                self.strokes_num[_id] = {
                    "x_points": _x_points,
                    "y_points": _y_points
                }

        # calculate center of x, y
        self.x_mean = np.mean(_x_pts)
        self.y_mean = np.mean(_y_pts)

        try:
            self.x_min = np.min(_x_pts)
            self.y_max = np.max(_y_pts)
        except:
            print("here")

        x_max = np.max(_x_pts)
        y_min = np.min(_y_pts)
        self.width = x_max - self.x_min
        self.height = self.y_max - y_min

    # stroke data number to string
    def update_stroke_data(self, selected_ids):
        for _id in self.strokes.keys():
            if _id in selected_ids:
                _new_stroke_data = []

                for _xx, _yy in zip(self.strokes_num[_id]["x_points"], self.strokes_num[_id]["y_points"]):
                    _new_stroke_data.append("%d %d" % (int(_yy), int(_xx)))

                self.strokes[_id] = ", ".join(_new_stroke_data)

    def re_arrange_stroke_value(self, selected_ids, padd):

        # strokke data string to number
        self.parse_stroke_data(selected_ids)

        min = [100000, 10000]
        # find min cord
        for _id in self.strokes.keys():
            if _id in selected_ids:
                _min_x = np.min(self.strokes_num[_id]["x_points"])
                _min_y = np.min(self.strokes_num[_id]["y_points"])

                if _min_x < min[0]:
                    min[0] = _min_x

                if _min_y < min[1]:
                    min[1] = _min_y

        # transformation
        _new_stroke_num = {}
        for _id in self.strokes.keys():
            if _id in selected_ids:
                # transformation
                _x_points, _y_points = transformation(x_points=self.strokes_num[_id]["x_points"],
                                                      y_points=self.strokes_num[_id]["y_points"],
                                                      angle_factor=0,
                                                      scale_factor=1,
                                                      translate_x_factor=padd - min[0],
                                                      translate_y_factor=padd - min[1],
                                                      center_x=self.x_mean,
                                                      center_y=self.y_mean)

                # save
                _new_stroke_num[_id] = {
                    "x_points": _x_points,
                    "y_points": _y_points
                }
        self.strokes_num = _new_stroke_num

        # update stroke string
        self.update_stroke_data(selected_ids)

    def transformation(self, selected_ids, angle_factor=0, scale_factor=1, tx_factor=0, ty_factor=0):
        # strokke data string to number
        self.parse_stroke_data(selected_ids)

        if self.height == 0 or self.width == 0:
            angle_factor = 0
        else:
            ratio = self.width / self.height
            if ratio < 1:
                ratio = 1/ratio
            angle_factor = angle_factor / ratio

        # transformation
        _new_stroke_num = {}
        for _id in self.strokes.keys():
            if _id in selected_ids:
                # transformation
                _x_points, _y_points = transformation(x_points=self.strokes_num[_id]["x_points"],
                                                      y_points=self.strokes_num[_id]["y_points"],
                                                      angle_factor=angle_factor,
                                                      translate_x_factor=tx_factor,
                                                      translate_y_factor=ty_factor,
                                                      scale_factor=scale_factor,
                                                      center_x=self.x_mean,
                                                      center_y=self.y_mean)

                # save
                _new_stroke_num[_id] = {
                    "x_points": _x_points,
                    "y_points": _y_points
                }

        self.strokes_num = _new_stroke_num

        # update stroke string
        self.update_stroke_data(selected_ids)

    def get_bound_box(self, selected_ids):
        # strokke data string to number
        self.parse_stroke_data(selected_ids)

        return self.x_min, self.y_max - self.height, self.width, self.height

    def get_sid_segid(self):
        # add segments
        max_seg = 0
        max_stroke = 0
        for _seg_name in self.segments:
            if ":" in _seg_name:
                max_seg = int(_seg_name.split(":")[-2])
            elif int(_seg_name) > max_seg:
                max_seg = int(_seg_name)
        for _st_name in self.strokes:
            if int(_st_name) > max_stroke:
                max_stroke = int(_st_name)
        return max_stroke, max_seg

    def update_sid_segid(self, max_sid, max_segid):
        # update strokes
        map_sid = {}
        _strokes_names = list(self.strokes.keys())
        for _st_name in _strokes_names:
            max_sid += 1
            _new_sid = "%d"%max_sid
            if _st_name == _new_sid:
                max_sid += 1
                _new_sid = "%d" % max_sid
            map_sid[_st_name] = _new_sid
            self.strokes[_new_sid] = self.strokes[_st_name]
            del self.strokes[_st_name]

        # update segments
        _segment_names = list(self.segments.keys())
        for _seg_name in _segment_names:
            max_segid += 1
            # update stroke id
            _new_strId = set([map_sid[ii] for ii in self.segments[_seg_name].strId if ii in map_sid] )
            self.segments[_seg_name].strId = _new_strId

            # update segment id
            _new_segid = "%d" % max_segid
            self.segments[_new_segid] = self.segments[_seg_name]
            del self.segments[_seg_name]

        return map_sid

    def swap_ink_object(self, ori_ids, new_ink_objs, new_gt):
        # assign new truth
        self.truth = new_gt

        # remove selected origin ids
        _strokes = {}
        for id in sorted(self.strokes.keys(), key=lambda x: float(x)):
            if id not in ori_ids:
                _strokes[id] = self.strokes[id]
        self.strokes = _strokes

        # remove selected origin segment
        _new_segments = {}
        for _seg_name in self.segments:
            _seg = self.segments[_seg_name]
            _str_ids = list(_seg.strId)
            if len(_str_ids):
                _str_id = _str_ids[0]
            else:
                continue
            if _str_id not in ori_ids and _seg not in _new_segments:
                _new_segments[_seg_name] = _seg
        self.segments = _new_segments

        # add new ink obj
        for _sid in new_ink_objs.strokes:
            self.strokes[_sid] = new_ink_objs.strokes[_sid]
        for _seg_id in new_ink_objs.segments:
            self.segments[_seg_id] = new_ink_objs.segments[_seg_id]

    def get_stroke_data(self, selected_ids):

        _x_pts = []
        _y_pts = []
        for _id in self.strokes.keys():
            if _id in selected_ids:
                _x_points = []
                _y_points = []
                # get real id
                _s_data = self.strokes[_id]
                _s_cords = _s_data.split(",")
                for _cord in _s_cords:
                    _term = _cord.split()
                    if "." in _s_data:
                        _x_points.append(int(float(_term[1])*100000))
                        _y_points.append(int(float(_term[0])*100000))
                    else:

                        _x_points.append(int(_term[1]))
                        _y_points.append(int(_term[0]))

                # TAKE CARE THIS. -1 for split between strokes
                _x_points.append(int(-1))
                _y_points.append(int(-1))
                _x_pts = _x_pts + _x_points
                _y_pts = _y_pts + _y_points

        # calculate center of x, y
        return _x_pts, _y_pts