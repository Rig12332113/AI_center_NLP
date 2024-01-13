from prompt_threading import prompts
import pandas as pd
import os 
import datetime

def main():
    start = datetime.datetime.now() 
    #os.environ['OPENAI_API_KEY'] = "..."
    prompt = prompts()
    
    docName = input("Input document name: ")
    writer = pd.ExcelWriter("檢核結果.xlsx")
    df = pd.DataFrame({"CheckItemName":[], "StandardValue":[], "最佳關聯段落":[], "最佳關聯段落相關片段":[], "檢核結果":[], "檢查過程":[]})
    data = pd.read_csv('./' + docName + '.csv',
                    usecols=["CheckItemName", "CheckRule", "StandardValue",
                             "規範序號_k=1", "規範章碼_k=1", "規範章名_k=1", "溯源頁碼_k=1", "關聯段落_k=1", 
                             "規範序號_k=2", "規範章碼_k=2", "規範章名_k=2", "溯源頁碼_k=2", "關聯段落_k=2", 
                             "規範序號_k=3", "規範章碼_k=3", "規範章名_k=3", "溯源頁碼_k=3", "關聯段落_k=3", 
                             "規範序號_k=4", "規範章碼_k=4", "規範章名_k=4", "溯源頁碼_k=4", "關聯段落_k=4", 
                             "規範序號_k=5", "規範章碼_k=5", "規範章名_k=5", "溯源頁碼_k=5", "關聯段落_k=5",
                             "規範序號_k=6", "規範章碼_k=6", "規範章名_k=6", "溯源頁碼_k=6", "關聯段落_k=6",
                             "規範序號_k=7", "規範章碼_k=7", "規範章名_k=7", "溯源頁碼_k=7", "關聯段落_k=7", 
                             "規範序號_k=8", "規範章碼_k=8", "規範章名_k=8", "溯源頁碼_k=8", "關聯段落_k=8", 
                             "規範序號_k=9", "規範章碼_k=9", "規範章名_k=9", "溯源頁碼_k=9", "關聯段落_k=9",
                             "規範序號_k=10", "規範章碼_k=10", "規範章名_k=10", "溯源頁碼_k=10", "關聯段落_k=10"]).values

    for i in range(data.shape[0]):
        content = prompt.divide_content(data[i, 0], data[i, 1])
        #print(content)
        best = prompt.ai_grading(content, data[i, :])
        #print(best)
        reply = prompt.ai_checking(data[i, 0], data[i, 2], best)
        #print(reply)
        new_row = {"CheckItemName":data[i, 0], "StandardValue":data[i, 2], "最佳關聯段落":best, "最佳關聯段落相關片段":reply["相關段落"], 
                   "檢核結果":reply["檢查結果"], "檢查過程":reply["檢查過程"]}
        df = df._append(new_row, ignore_index = True)

    df.to_excel(writer, index=False)
    writer._save()
    end = datetime.datetime.now()
    print("執行時間: " + str(end - start))


if __name__ == '__main__':
    main()