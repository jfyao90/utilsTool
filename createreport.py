
import tempfile

from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, LongTable, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from io import BytesIO

pdfmetrics.registerFont(TTFont('SimSun', './font/SimSun.ttf'))  # 默认不支持中文，需要注册字体
pdfmetrics.registerFont(TTFont('SimSunBd', './font/SimSun-bold.ttf'))
registerFontFamily('SimSun', normal='SimSun', bold='SimSunBd', italic='VeraIt', boldItalic='VeraBI')

stylesheet = getSampleStyleSheet()  # 获取样式集

# 获取reportlab自带样式
Normal = stylesheet['Normal']
BodyText = stylesheet['BodyText']
Italic = stylesheet['Italic']
Title = stylesheet['Title']
Heading1 = stylesheet['Heading1']
Heading2 = stylesheet['Heading2']
Heading3 = stylesheet['Heading3']
Heading4 = stylesheet['Heading4']
Heading5 = stylesheet['Heading5']
Heading6 = stylesheet['Heading6']
Bullet = stylesheet['Bullet']
Definition = stylesheet['Definition']
Code = stylesheet['Code']

# 自带样式不支持中文，需要设置中文字体，但有些样式会丢失，如斜体Italic。有待后续发现完全兼容的中文字体
Normal.fontName = 'SimSun'
Italic.fontName = 'SimSun'
BodyText.fontName = 'SimSun'
Title.fontName = 'SimSunBd'
Heading1.fontName = 'SimSun'
Heading2.fontName = 'SimSun'
Heading3.fontName = 'SimSun'
Heading4.fontName = 'SimSun'
Heading5.fontName = 'SimSun'
Heading6.fontName = 'SimSun'
Bullet.fontName = 'SimSun'
Definition.fontName = 'SimSun'
Code.fontName = 'SimSun'

# 添加自定义样式
stylesheet.add(
    ParagraphStyle(name='body',
                   fontName="SimSun",
                   fontSize=10,
                   textColor='black',
                   leading=20,  # 行间距
                   spaceBefore=0,  # 段前间距
                   spaceAfter=0,  # 段后间距
                   leftIndent=0,  # 左缩进
                   rightIndent=0,  # 右缩进
                   firstLineIndent=20,  # 首行缩进，每个汉字为10
                   alignment=TA_JUSTIFY,  # 对齐方式

                   # bulletFontSize=15,       #bullet为项目符号相关的设置
                   # bulletIndent=-50,
                   # bulletAnchor='start',
                   # bulletFontName='Symbol'
                   )
)

stylesheet.add(ParagraphStyle(name='label', fontName ='SimSun',fontSize=8, textColor='black',spaceBefore=0,  leftIndent=150,  alignment=TA_JUSTIFY))

body = stylesheet['body']
label = stylesheet['label']

story = []

# 段落
content1 = "原始数据中不同类别记录的分布情况，如下图所示，其中class-0表示正常行为，class-1表示异常行为。"  \
            "<br/><img src='./distribution.png' width=450 height=230 valign='top'/><br/>"  \
           "<br/><br/><br/><br/>"   \
            "<br/><br/><br/><br/><br/><br/>"

content10 = "<br/>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp  图1 原始数据中不同类别的分布情况<br/>"   \
            "<br/><br/><br/>"

content11 = "从下图可以看出，该数据集不存在缺失值，因此不需作缺失值处理。"  \
           "<br/><img src='./Missingno.png' width=450 height=230 "  \
            "valign='top'/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>"

content110 = "<br/>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp  图2 原始数据记录的缺失值情况<br/>"

content12 = "下表给出了该数据集的基本统计信息。"  \
           "<br/><img src='./describe.png' width=450 height=700 "  \
            "valign='top'/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>" \
            "<br/><br/><br/><br/><br/><br/><br/> <br/><br/><br/><br/><br/><br/><br/>" \
            "<br/><br/><br/><br/><br/>"

content2 = "从图3可以看出，在信用卡被盗刷的事件中，部分变量之间的相关性更明显。其中变量V1、V2、V3、"  \
           "V4、V5、V6、V7、V9、V10、V11、V12、V14、V16、V17和V18以及V19之间的变化在信用卡被盗"  \
           "刷的样本中呈性一定的规律。<br/><img src='./NormalFraud.png' width=450 height=230 "  \
            "valign='top'/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>"

content20 = "<br/>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp  图3 各变量与信用卡行为间的关系<br/><br/>"

content3 = "从图4可以看出，信用卡被盗刷发生的金额与信用卡正常用户发生的金额相比呈现散而小的特点，"  \
            "这说明信用卡盗刷者为了不引起信用卡卡主的注意，更偏向选择小金额消费。"  \
           "<br/><img src='./Transactions.png' width=450 height=230 valign='top'/>"  \
           "<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>"

content30 = "<br/>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp  图4 盗刷交易、交易金额和交易次数的关系<br/>"

content4 = "从图5可以看出，在信用卡被盗刷样本中，离群值发生在客户使用信用卡消费更低频的时间段。"  \
            "信用卡被盗刷数量案发最高峰在第一天上午11点达到43次，其余发生信用卡被盗刷案发时间在晚上时"  \
            "间11点至第二早上9点之间，说明信用卡盗刷者为了不引起信用卡卡主注意，更喜欢选择信用卡"  \
            "卡主睡觉时间和消费频率较高的时间点作案；同时，信用卡发生被盗刷的最大值也就只有2,125.87美元。"  \
           "<br/><img src='./Amount.png' width=450 height=230 valign='top'/><br/><br/><br/><br/><br/>"  \
            "<br/><br/><br/><br/><br/><br/>"

content40 = "<br/>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp  图5 盗刷交易、交易金额和交易时间的关系<br/><br/><br/>"

content5 = "下图是各个变量在信用卡被盗刷和信用卡正常情况下的不同分布情况，我们将选择在不同信用卡状态下的分布"  \
            "有明显区别的变量。因此剔除变量V8、V13 、V15 、V20 、V21 、V22、 V23 、V24 、V25 、"  \
            "V26 、V27 和V28变量。这也与我们开始用相关性图谱观察得出结论一致。同时剔除变量Time，保留离散程度更小的Hour变量。"  \
           "<br/><img src='./feature0.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"   \
            "<br/><br/>"

content6 =  "<br/><img src='./feature1.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"   \
            "<br/><br/><br/>"
content61 = "<br/><img src='./feature2.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"   \
            "<br/><br/><br/>"
content62 = "<br/><img src='./feature3.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"   \
            "<br/><br/><br/>"
content63 = "<br/><img src='./feature4.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"


content64 = "<br/><img src='./feature5.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"   \
            "<br/><br/><br/>"
content65 = "<br/><img src='./feature6.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"   \
            "<br/><br/><br/>"
content66 = "<br/><img src='./feature7.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"   \
            "<br/><br/><br/>"
content67 = "<br/><img src='./feature8.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"



content68 = "<br/><img src='./feature9.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"   \
            "<br/><br/><br/>"
content69 = "<br/><img src='./feature10.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"   \
            "<br/><br/><br/>"
content610 = "<br/><img src='./feature11.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"   \
            "<br/><br/><br/>"
content611 =  "<br/><img src='./feature12.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"



content612 = "<br/><img src='./feature13.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"  \
            "<br/><br/><br/>"
content613 = "<br/><img src='./feature14.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"   \
            "<br/><br/><br/>"
content614 = "<br/><img src='./feature15.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"   \
            "<br/><br/><br/>"
content615 = "<br/><img src='./feature16.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"



content616 = "<br/><img src='./feature17.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"   \
            "<br/><br/><br/>"
content617 = "<br/><img src='./feature18.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"  \
            "<br/><br/><br/>"
content618 = "<br/><img src='./feature19.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"   \
            "<br/><br/><br/>"
content619 = "<br/><img src='./feature20.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"


content620 = "<br/><img src='./feature21.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"   \
            "<br/><br/><br/>"
content621 = "<br/><img src='./feature22.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"   \
            "<br/><br/><br/>"
content622 = "<br/><img src='./feature23.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"   \
            "<br/><br/><br/>"
content623 = "<br/><img src='./feature24.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"


content624 = "<br/><img src='./feature25.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"   \
            "<br/><br/><br/>"
content625 = "<br/><img src='./feature26.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"   \
            "<br/><br/><br/>"
content626 = "<br/><img src='./feature27.png' width=450 height=150 valign='top'/><br/><br/><br/><br/><br/>"   \
            "<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>"



content71 = "变量V1、V2、V3、V4、V5、V6、V7、V9、V10、V11、V12、V14、V16、V17、V18以及V19的重要性分析如下图所示。"  \
           "<br/><img src='./importances.png' width=450 height=230 valign='top'/><br/><br/><br/><br/><br/>"   \
            "<br/><br/><br/><br/><br/><br/><br/><br/>"



content7 = "不同阈值下的混淆矩阵如下图所示。" \
            "<br/><img src='./thresholds.png' width=450 height=230 valign='top'/><br/><br/><br/><br/><br/>"   \
            "<br/><br/><br/><br/><br/><br/><br/>"  \
           "&nbsp&nbsp&nbsp&nbsp  precision和recall是一组矛盾的变量。从混淆矩阵可以看到，阈值越小，recall值越大，模型能找" \
           "出信用卡被盗刷的数量也就更多，但换来的代价是误判的数量也较大。随着阈值的提高，recall值逐渐降低，" \
           "precision值也逐渐提高，误判的数量也随之减少。通过调整模型阈值，控制模型反信用卡欺诈的力度，" \
           "若想找出更多的信用卡被盗刷就设置较小的阈值，反之，则设置较大的阈值。" \
           "实际业务中，阈值的选择取决于公司业务边际利润和边际成本的比较；当模型阈值设置较小的值，" \
           "确实能找出更多的信用卡被盗刷的持卡人，但随着误判数量增加，不仅加大了贷后团队的工作量，" \
           "也会降低误判为信用卡被盗刷客户的消费体验，从而导致客户满意度下降，如果某个模型阈值能让业务的边际利润" \
           "和边际成本达到平衡时，则该模型的阈值为最优值。"   \
            "<br/><br/>"  \
           "&nbsp&nbsp&nbsp&nbsp  模型ROC曲线。"  \
           "<br/><img src='./ROC.png' width=450 height=230 valign='top'/><br/><br/><br/><br/><br/>"  \
            "<br/><br/><br/><br/><br/><br/><br/><br/>"   \


content8 = "模型最优性能如下图所示。"  \
           "<br/><img src='./thresholds.png' width=450 height=230 valign='top'/><br/><br/><br/><br/><br/>"  
#
# content1 = "<para><u color='red'><font fontSize=13>区块链</font></u>是分布式数据存储、<strike color='red'>点对点传输</strike>、共识机制、" \
#            "<font color='red' fontSize=13>加密算法</font>等计算机技术的<font name='SimSunBd'>新型应用模式</font>。<br/>" \
#            "&nbsp&nbsp<a href='www.baidu.com' color='blue'>区块链（Blockchain）</a>，" \
#            "是比特币的一个重要概念，它本质上是一个去中心化的数据库，同时作为比特币的底层技术，是一串使用密码学方法相关联产生的" \
#            "数据块，每一个数据块中包含了一批次比特币网络交易的信息，用于验证其信息的有效性（防伪）和生成下一个区块 [1]。</para>"
#
# content2 = "区块链起源于比特币，2008年11月1日，一位自称中本聪(SatoshiNakamoto)的人发表了《比特币:一种点对点的电子现金系统》" \
#            "一文 [2]  ，阐述了基于P2P网络技术、加密技术、时间戳技术、区块链技术等的电子现金系统的构架理念，这标志着比特币的诞生" \
#            "。两个月后理论步入实践，2009年1月3日第一个序号为0的创世区块诞生。几天后2009年1月9日出现序号为1的区块，并与序号为" \
#            "0的创世区块相连接形成了链，标志着区块链的诞生 [5]  。<br/><img src='./1.png' width=180 height=100 valign='top'/><br/><br/><br/><br/><br/>"
#
# content3 = "2008年由中本聪第一次提出了区块链的概念 [2]  ，在随后的几年中，区块链成为了电子货币比特币" \
#            "的核心组成部分：作为所有交易的公共账簿。通过利用点对点网络和分布式时间戳服务器，区块链数据库能够进行自主管理。为比特币而发明的区块链使它成为" \
#            "第一个解决重复消费问题的数字货币。比特币的设计已经成为其他应用程序的灵感来源。<br/>&nbsp&nbsp 2014年，区块链2.0成为一个关于去中心" \
#            "化区块链数据库的术语。对这个第二代可编程区块链，经济学家们认为它是一种编程语言，可以允许用户写出更精密和智能的协议 " \
#            "[7]  。因此，当利润达到一定程度的时候，就能够从完成的货运订单或者共享证书的分红中获得收益。区块链2.0技术跳过了交易" \
#            "和“价值交换中担任金钱和信息仲裁的中介机构”。它们被用来使人们远离全球化经济，使隐私得到保护，使人们“将掌握的信息兑换" \
#            "成货币”，并且有能力保证知识产权的所有者得到收益。第二代区块链技术使存储个人的“永久数字ID和形象”成为可能，并且对“潜在" \
#            "的社会财富分配”不平等提供解决方案 [8]  。<br/>&nbsp&nbsp 2016年1月20日，中国人民银行数字货币研讨会宣布对数字货币研究取得阶段性成果。" \
#            "会议肯定了数字货币在降低传统货币发行等方面的价值，并表示央行在探索发行数字货币。中国人民银行数字货币研讨会的表达大大" \
#            "增强了数字货币行业信心。这是继2013年12月5日央行五部委发布关于防范比特币风险的通知之后，第一次对数字货币表示明确的态度" \
#            "。 [9] <br/>&nbsp&nbsp 2016年12月20日，数字货币联盟——中国FinTech数字货币联盟及FinTech研究院正式筹建 [10]  。<br/>&nbsp&nbsp如今，比特币仍是" \
#            "数字货币的绝对主流，数字货币呈现了百花齐放的状态，常见的有bitcoin、litecoin、dogecoin、dashcoin，除了货币的应用" \
#            "之外，还有各种衍生应用，如以太坊Ethereum、Asch等底层应用开发平台以及NXT，SIA，比特股，MaidSafe，Ripple等行业应用 [11]  。"

# Table 表格

image = Image('./1.png')
image.drawWidth = 160
image.drawHeight = 100
table_data = [['year我是标题行，\n\n比较特殊，不能上下居中\n', '我的背景色被绿了', '我是标题，我比别人大\n'],
              ['2017\n我是换行符，\n单元格中的字符串只能用我换行', '3', '12'],
              [Paragraph('指定了列宽，可以在单元格中嵌入paragraph进行自动换行，不信你看我', body), '4', '13'],
              ['2017', '5', '我们是表格'],
              ['2017', '我是伪拆分单元格，\n通过合并前hou两个兄弟得到', '15'],
              ['2018', '7', '16'],
              [Paragraph(content1, body), '8', [image, Paragraph('这样我可以在一个单元格内同时显示图片和paragraph', body)]],
              ['2018', '我们被合并了，合并的值为右上角单元格的值', '18'],
              ['我被绿了', '10', '19'],
              ]
table_style = [
    ('FONTNAME', (0, 0), (-1, -1), 'SimSun'),  # 字体
    ('FONTSIZE', (0, 0), (-1, 0), 15),  # 第一行的字体大小
    ('FONTSIZE', (0, 1), (-1, -1), 10),  # 第二行到最后一行的字体大小
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # 所有表格左右中间对齐
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # 所有表格上下居中对齐

    ('SPAN', (-2, -2), (-1, -1)),  # 合并
    ('SPAN', (0, 4), (0, 5)),  # 合并
    ('SPAN', (2, 4), (2, 5)),  # 合并
    ('BACKGROUND', (0, 0), (-1, 0), colors.green),  # 设置第一行背景颜色
    ('TEXTCOLOR', (0, -1), (0, -1), colors.green),  # 设置表格内文字颜色
    ('GRID', (0, 0), (-1, -1), 0.1, colors.black),  # 设置表格框线为灰色，线宽为0.1
]
table = Table(data=table_data, style=table_style, colWidths=180)

story.append(Paragraph("信用卡欺诈行为分析报告", Title))
story.append(Paragraph("<seq id='spam'/>.原始数据分布情况", Heading2))
story.append(Paragraph(content1, body))
story.append(Paragraph(content10, label))
story.append(Paragraph("<seq id='spam'/>.数据缺失情况分析", Heading2))
story.append(Paragraph(content11, body))
story.append(Paragraph(content110, label))
story.append(Paragraph("<seq id='spam'/>.基本统计信息分析", Heading2))
story.append(Paragraph(content12, body))
story.append(Paragraph("<seq id='spam'/>.信用卡正常用户与被盗刷用户之间的区别", Heading2))
story.append(Paragraph(content2, body))
story.append(Paragraph(content20, label))
story.append(Paragraph("<seq id='spam'/>.盗刷交易、交易金额和交易次数的关系", Heading2))
story.append(Paragraph(content3, body))
story.append(Paragraph(content30, label))
story.append(Paragraph("<seq id='spam'/>.盗刷交易、交易金额和交易时间的关系", Heading2))
story.append(Paragraph(content4, body))
story.append(Paragraph(content40, label))
story.append(Paragraph("<seq id='spam'/>.特征V1~V28对信用卡行为的影响情况分析", Heading2))
story.append(Paragraph(content5, body))
story.append(Paragraph(content6, body))
story.append(Paragraph(content61, body))
story.append(Paragraph(content62, body))
story.append(Paragraph(content63, body))
story.append(Paragraph(content64, body))
story.append(Paragraph(content65, body))
story.append(Paragraph(content66, body))
story.append(Paragraph(content67, body))
story.append(Paragraph(content68, body))
story.append(Paragraph(content69, body))
story.append(Paragraph(content610, body))
story.append(Paragraph(content611, body))
story.append(Paragraph(content612, body))
story.append(Paragraph(content613, body))
story.append(Paragraph(content614, body))
story.append(Paragraph(content615, body))
story.append(Paragraph(content616, body))
story.append(Paragraph(content617, body))
story.append(Paragraph(content618, body))
story.append(Paragraph(content619, body))
story.append(Paragraph(content620, body))
story.append(Paragraph(content621, body))
story.append(Paragraph(content622, body))
story.append(Paragraph(content623, body))
story.append(Paragraph(content624, body))
story.append(Paragraph(content625, body))
story.append(Paragraph(content626, body))
story.append(Paragraph("<seq id='spam'/>.特征重要性分析", Heading2))
story.append(Paragraph(content71, body))
story.append(Paragraph("<seq id='spam'/>.模型优化与评估", Heading2))
story.append(Paragraph(content7, body))
story.append(Paragraph("<seq id='spam'/>.模型性能", Heading2))
story.append(Paragraph(content8, body))
# story.append(table)

# bytes
# buf = BytesIO()
# doc = SimpleDocTemplate(buf, encoding='UTF-8')
# doc.build(story)
# print(buf.getvalue().decode())

# file
doc = SimpleDocTemplate('hello.pdf')
doc.build(story)