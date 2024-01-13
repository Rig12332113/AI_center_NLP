"""
Prompts class contains prompts and functions for LLM searching.
Major functions include divide_content, ai_grading and ai_checking.
"""

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
import threading

class prompts:
    def __init__(self):
        self.llm = ChatOpenAI(model = "gpt-4", temperature=0.2)
        self.divide_content_template = """
                            請先閱讀分項例子：
                            混凝土澆置 - 鋼套管埋入混凝土中大於 2 m

                            該表單項目包括以下細項：
                            1. 混凝土澆置時鋼套管需埋入混凝土中
                            2. 鋼套管埋入深度需大於2m

                            你現在是一名工程師，請參考上述例子將以下提供的表單項目依照內容分為細項。
                            表單項目：{CheckItemName} - {CheckRule}
                            表單項目結束

                            請只回覆細項，不要加任何多餘文字，限制字數80字
                            """
        self.grading_template = """
                    請先閱讀評分例子：
                    範例一：

                    關鍵段落：
                    (2) 混凝土澆置過程承包商應記錄混凝土澆置量及套管內混凝土高度，套管
                    底端應低於混凝土面至少 2m，以免孔壁崩塌，致使基樁斷面減少或混凝土品質受損。  
                    關鍵段落結束
                    細項：
                    1. 混凝土澆置時鋼套管需埋入混凝土中 -- 100分
                    2. 鋼套管埋入深度需大於2m -- 100分
                    整體評分： (100 + 100) / 2 = 100分。

                    範例二：

                    關鍵段落：
                    本款第 (2)目原列條文修訂如下： 『 (2) 混凝土運送時應保持品質均勻，避免不當之材料析離或坍度損失。除另
                    有規定外，混凝土自拌和完成後至工地開始卸料之時間規定如下：  
                    A. 運送途中保持攪動者不得超過 90分鐘。  B. 途中未加攪動者不得超過 30分鐘。』 (補說 )
                    細項：
                    1. 混凝土澆置及運輸的全部時間需小於120分鐘 -- 0分
                    2. 混凝土澆置的停頓時間需小於30分鐘 -- 0分
                    整體評分： (0 + 0) / 2 = 0分。

                    你現在是一名工程師，請參考範例將下列關鍵段落依細項評分，最後給出0至100的整體分數。
                    （0為規範內沒有提及，100為規範有條文提及）
                    細項：{Content}
                    細項結束
                    關鍵段落：{KeyParagraph}
                    關鍵段落結束

                    {Reply}
                    """
        self.checking_template = """
                    請先閱讀下列檢核範例：

                    規範內容：
                    A.將模板或構造物包圍加溫，使其內之混凝土及氣溫保持在 13℃以上。
                    完成澆置之混凝土應維持該溫度 7天。 
                    B.於混凝土養護期間加溫時，其周圍之相對溼度應維持不低於 40%。火
                    爐、烤板或加熱器應妥為佈設，使熱量均勻分佈。燃燒之廢氣體應排
                    至包圍體外部(115-3050-混凝土基本材料及施工一般要求(10702修正)(監R)-12)
                    (1) 周圍溫度超過 32℃以上時，應於澆置混凝土前，將模板及鋼筋等以水加
                    以冷卻，降溫至 32℃以下，方可開始澆置混凝土。 (115-3050-混凝土基本材料及施工一般要求(10702修正)(監R)-13)

                    表單內容：
                    ★混凝土澆置-溫度 - 保持在13℃以上，32℃以下。

                    相關段落：
                    將模板或構造物包圍加溫，使其內之混凝土及氣溫保持在 13℃以上。
                    (115-3050-混凝土基本材料及施工一般要求(10702修正)(監R)-12)
                    周圍溫度超過 32℃以上時，應於澆置混凝土前，將模板及鋼筋等以水加
                    以冷卻，降溫至 32℃以下，方可開始澆置混凝土。 
                    (115-3050-混凝土基本材料及施工一般要求(10702修正)(監R)-13)

                    檢查結果：
                    T符合規範。

                    你現在是一名工程師，請參考範例擷取規範內容中與表單內容有關之相關段落，並檢查表單內容是否符合規範之相關段落。
                    請依格式提供擷取的相關段落、檢查過程與檢查結果。
                    （F為不符合規範，T為符合規範）
                    規範內容: {Best}
                    規範內容結束
                    表單內容: {CheckItemName} - {StandardValue}
                    表單內容結束

                    {Reply}
                    """
    
    #Extract grade from reply
    def _extract_number(self, input_number):
        comma = False
        digit_after_comma = 1
        number = 0
        for i in range(len(input_number)):
            if input_number[i].isdigit() and comma == False:
                number = number * 10 + int(input_number[i])
            elif input_number[i] == ".":
                comma = True
            elif input_number[i].isdigit() and comma == True:
                number = number + int(input_number[i]) / 10 ** digit_after_comma
                digit_after_comma += 1 
        return number
    
    # Grade one key paragraph
    def _grade(self, content, keyParagraph, grades, index):
        itemAndgrade_schema = ResponseSchema(name = "細項與細項評分", description = "內容")
        overallGrade_schema = ResponseSchema(name = "整體分數", description = "分數")
        response_schemas = [itemAndgrade_schema, overallGrade_schema]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        grading_templateprompt = PromptTemplate(input_variables = ["Content", "KeyParagraph", "Reply"], template = self.grading_template) 
        grading_chain = LLMChain(llm = self.llm, prompt = grading_templateprompt)
        while True:
            try:
                reply = grading_chain({"Content":content['text'], "KeyParagraph":keyParagraph, "Reply":format_instructions})
                formatted_reply = output_parser.parse(reply['text'])
            except:
                continue
            break
        grades[index] = (self._extract_number(formatted_reply['整體分數']))
     
    # Divide content (細項拆分)
    # Input: one checkItemName and one CheckRule; Output: devided content
    def divide_content(self, checkItemName, checkRule):
        content_templateprompt = PromptTemplate(input_variables = ['CheckItemName', 'CheckRule'], template = self.divide_content_template)
        content_chain = LLMChain(llm = self.llm, prompt = content_templateprompt)
        content = content_chain({"CheckItemName":checkItemName, "CheckRule":checkRule})
        return content

    # Grade top key paragraphs by content and reply the best paragraph（LLM top 10篩選最佳段落）
    # Input: devided content and data extracted from check form; Output: best paragraphs
    def ai_grading(self, content, data):
        paragraph_count = (data.shape[0] - 3) // 5
        best_paragraph_grade = 0
        best_source = []
        best_paragraph = [] 
        grades = [i for i in range (paragraph_count)]
        threads = []
        for i in range(paragraph_count):
            threads.append(threading.Thread(target=self._grade, args=(content, data[7 + 5 * i], grades, i)))
            threads[i].start()
        for i in range(paragraph_count): 
            threads[i].join()
        #找最佳關鍵段落
        for i in range(len(grades)):
            if grades[i] > best_paragraph_grade:
                best_source = []
                best_paragraph = []
                best_paragraph_grade = grades[i]
            if grades[i] >= best_paragraph_grade:
                source = (data[3 + 5 * i], data[4 + 5 * i], data[5 + 5 * i], data[6 + 5 * i])
                keyparagraph = data[7 + 5 * i]
                best_source.append(source)
                best_paragraph.append(keyparagraph + "(" + str(source[0]) + "-" + str(source[1]) + "-" + str(source[2]) + "-" + str(source[3]) + ")")
        
        if best_paragraph_grade == 0:
            response = "無相關規範\n"
        else:
            response = best_paragraph

        return response
    
    # Extract part related to checkRule and check the standard by the best paragraphs（找最佳段落中相關片段並檢核查驗標準）
    # Input: one checkRule, one standardValue and best paragraphs; Output: contains related parts, reason and result.
    def ai_checking(self, checkItemName, standardValue, best_paragraph):
        related_paragraoh_schema = ResponseSchema(name = "相關段落", description = "段落內容")
        reason_schema = ResponseSchema(name = "檢查過程", description = "判斷過程")
        rule_check_schema = ResponseSchema(name = "檢查結果", description = "分數")
        response_schemas = [related_paragraoh_schema, reason_schema, rule_check_schema]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        best = ""
        for i in range(len(best_paragraph)):
            best += best_paragraph[i]
        checking_templateprompt = PromptTemplate(input_variables = ["Best", "CheckItemName", "StandardValue", "Reply"], template = self.checking_template) 
        grading_chain = LLMChain(llm = self.llm, prompt = checking_templateprompt)
        success = False
        times = 0
        while success == False and times < 3:
            try:
                reply = grading_chain({"Best":best, "CheckItemName": checkItemName, "StandardValue": standardValue, "Reply":format_instructions})
                formatted_reply = output_parser.parse(reply['text'])
                if formatted_reply["檢查結果"] == "T" or formatted_reply["檢查結果"] == "F":
                    success = True
            except:
                times += 1

        return formatted_reply
