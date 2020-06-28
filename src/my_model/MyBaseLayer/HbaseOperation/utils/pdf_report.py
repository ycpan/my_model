from reportlab.lib.units import mm, cm
from reportlab.lib.pagesizes import A4, portrait, landscape
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfdoc
from reportlab.lib import colors
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase.pdfdoc import PDFImageXObject
from reportlab.lib.utils import _digester
from reportlab.pdfbase import pdfutils
from reportlab.graphics.charts.piecharts import Pie

import reportlab.pdfbase.ttfonts
import time
import io


class PDFImageObject(PDFImageXObject):
    def __init__(self, image_data, source=None, mask=None):
        self.mask = mask
        self.loadImageFromA85(image_data)

    def loadImageFromA85(self, image_data):
        IMG = []
        if not image_data:
            return self.loadImageFromSRC(IMG[0])
        image_data = [s.strip() for s in image_data]
        words = image_data[1].split()
        self.width, self.height = (int(words[1]), int(words[3]))
        self.colorSpace = {'/RGB': 'DeviceRGB', '/G': 'DeviceGray', '/CMYK': 'DeviceCMYK'}[words[7]]
        self.bitsPerComponent = 8
        self._filters = 'ASCII85Decode', 'FlateDecode'  # 'A85','Fl'
        if IMG:
            self._checkTransparency(IMG[0])
        elif self.mask == 'auto':
            self.mask = None
        self.streamContent = ''.join(image_data[3:-1])


class PDFReport(canvas.Canvas):
    def __init__(self, file_path):
        super(PDFReport, self).__init__(file_path)
        self.font_path = None
        self.font_size = None
        self.all_set_font = False
        self.name = "PDFReport"
        self.images = 0
        self.last_y = 29.7 * cm
        self.width, self.height = A4
        reportlab.pdfbase.pdfmetrics.registerFont(
            reportlab.pdfbase.ttfonts.TTFont('my_font', '/usr/share/fonts/truetype/arphic-gkai00mp/gkai00mp.ttf'))

        reportlab.lib.styles.ParagraphStyle.defaults['wordWrap'] = "CJK"
    def set_head(self, headtext):
        # A4 210 x 297 mm
        # self.setFont("Helvetica-Bold", 11.5)
        # 向一张pdf页面上写string
        self.writeString(1 * mm, 11.5 * mm, headtext)
        # 画一个矩形，并填充为黑色
        self.rect(1 * mm, 11.3 * mm, 6.5 * mm, 0.08 * mm, fill=1)
        # 画一条直线
        self.line(1 * mm, 11 * mm, 7.5 * mm, 11 * mm)

    def set_fonts(self, all_set=True, font_size=12, font_path=None):
        """

        :param all_set:
        :param font_size:
        :param font_path:
        :return:
        using:
        self.set_fonts('my_font', font_size)
        self.drawString(100, 300, u'宋体宋体')
        """
        self.all_set_font = all_set
        self.font_size = font_size
        self.font_path = font_path
        if font_path is not None:
            reportlab.pdfbase.pdfmetrics.registerFont(
                reportlab.pdfbase.ttfonts.TTFont('my_font', font_path))
        self.setFont('my_font', font_size)

        # self.drawString(10 * cm, self.last_y - 5 * cm, '宋体宋体')
        # self.drawString(10 * cm, self.last_y - 6 * cm, u'宋体1')
    def drawImage(self, image, x1, y1, x2=None, y2=None):  # Postscript Level2 version
        # select between postscript level 1 or level 2
        diff = self.last_y - y2
        if diff >= 10 * mm:
            self.last_y = diff
        else:
            self.showPage()
            self.last_y = self.last_y - y2
        y1 = self.last_y
        x, y = super(PDFReport, self).drawImage(image, x1, y1, x2, y2)
        self.last_y = y
        # self.images += 1

    def data_to_img(self, image_data, x, y, width=None, height=None, mask=None,
                    preserveAspectRatio=False, anchor='c'):
        """
        :param image_data:
        image_data can get from img_to_data
        :param x:
        :param y:
        :param width:
        :param height:
        :param mask:
        :param preserveAspectRatio:
        :param anchor:
        :return:
        """
        diff = self.last_y - height
        if diff >= 10 * mm:
            self.last_y = diff
        else:
            self.showPage()
            self.last_y = self.last_y - height
            # y1 = self.last_y
        # x, y = super(PDFReport, self).drawImage(image, x1, y1, x2, y2)
        # self.last_y = y
        y = self.last_y

        self._currentPageHasImages = 1
        name = _digester(str(time.time()))
        reg_name = self._doc.getXObjectName(name)
        img_obj = PDFImageObject(image_data)
        img_obj.name = name
        self._setXObjects(img_obj)
        self._doc.Reference(img_obj, reg_name)
        self._doc.addForm(name, img_obj)
        s_mask = getattr(img_obj, '_smask', mask)
        if s_mask:  # set up the softmask obtained above
            m_reg_name = self._doc.getXObjectName(s_mask.name)
            m_img_obj = self._doc.idToObject.get(m_reg_name, None)
            if not m_img_obj:
                self._setXObjects(s_mask)
                img_obj.smask = self._doc.Reference(s_mask, m_reg_name)
            else:
                img_obj.smask = pdfdoc.PDFObjectReference(m_reg_name)
            del img_obj._smask

        self.saveState()
        self.translate(x, y)
        self.scale(width, height)
        self._code.append("/%s Do" % reg_name)
        self.restoreState()
        #
        # # track what's been used on this page
        self._formsinuse.append(name)

        return (img_obj.width, img_obj.height)

    @staticmethod
    def img_to_data(canvas):
        """
        :param canvas:
        fig = plt.figure()
        plt.plot(x, y)
        canvas = fig.canvas
        :return:
        img list string that pdf can used
        """
        buffer = io.BytesIO()
        canvas.print_png(buffer)

        buffer_reader = io.BufferedReader(buffer)
        return pdfutils.makeA85Image(buffer_reader, IMG=[], detectJpeg=True)

    def showPage(self):
        super(PDFReport, self).showPage()
        self.images = 0
        self.last_y = 297 * mm
        if self.all_set_font:
            self.set_fonts(True, self.font_size, self.font_path)

    def writeString(self, x, y, text, mode=None, charSpace=0):

        diff = self.last_y - 10 * mm
        if diff >= 10 * mm:
            self.last_y = diff
        else:
            self.showPage()
            self.last_y = self.last_y - 10 * mm
        y = self.last_y
        super(PDFReport, self).drawString(x, y, text, mode, charSpace)

    def add_table(self, x, y, df_data, **arg):
        """
        
        :param df_data: 
        :param arg: the same as Table
        :return: 
        for example:
        ts = [
              ('FONTNAME',(0,0),(-1,-1),'msyh'),#字体
              ('FONTSIZE',(0,0),(-1,-1),6),#字体大小
              ('SPAN',(0,0),(3,0)),#合并第一行前三列
              ('BACKGROUND',(0,0),(-1,0), colors.lightskyblue),#设置第一行背景颜色
              ('SPAN',(-1,0),(-2,0)), #合并第一行后两列
              ('ALIGN',(-1,0),(-2,0),'RIGHT'),#对齐
              ('VALIGN',(-1,0),(-2,0),'MIDDLE'),  #对齐
              ('LINEBEFORE',(0,0),(0,-1),0.1,colors.grey),#设置表格左边线颜色为灰色，线宽为0.1
              ('TEXTCOLOR',(0,1),(-2,-1),colors.royalblue),#设置表格内文字颜色
              ('GRID',(0,0),(-1,-1),0.5,colors.red),#设置表格框线为红色，线宽为0.5
              ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
              ('LINEABOVE', (0, 0), (-1, 0), 1, colors.purple),
              ('LINEBELOW', (0, 0), (-1, 0), 1, colors.purple),
              ('FONT', (0, 0), (-1, 0), 'Times-Bold'),
              ('LINEABOVE', (0, -1), (-1, -1), 1, colors.purple),
              ('LINEBELOW', (0, -1), (-1, -1), 0.5, colors.purple, 1, None, None, 4, 1),
              ('LINEBELOW', (0, -1), (-1, -1), 1, colors.red),
              ('FONT', (0, -1), (-1, -1), 'Times-Bold'),
              ('BACKGROUND', (1, 1), (-2, -2), colors.green),
              ('TEXTCOLOR', (0, 0), (1, -1), colors.red)]
        cols=[5.05 * cm, 4.7 * cm, 5.0 * cm]
        add_table(list_df, style=ts, colWidths=cols)     
        """
        self.last_y =y - len(df_data) * 8 * mm
        if self.last_y <= 10 * mm:
            self.showPage()
            self.last_y = self.last_y - len(df_data) * 8 * mm
        y = self.last_y
        # def coord(x, y, unit=1):
        #     x, y = x * unit, self.height - y * unit
        #     return x, y

        list_df = [df_data.columns[:, ].values.astype(str).tolist()] + df_data.values.tolist()

        table = Table(list_df, **arg)
        table.setStyle(TableStyle([
            ('INNERGRID', (0, 0), (-1, -1), 0.50, colors.black),
            ('BOX', (0, 0), (-1, -1), 0.50, colors.black),
            ('FONTNAME',(0,0),(-1,-1),'my_font'),#字体
            # ('FONTSIZE',(0,0),(-1,-1),6),#字体大小
        ]))

        # c = canvas.Canvas("a.pdf", pagesize=A4)
        table.wrapOn(self, self.width, self.height)
        # table.drawOn(self, *coord(1.8, 9.6, cm))
        table.drawOn(self, x, y)
