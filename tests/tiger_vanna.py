import os

from vanna.anthropic.anthropic_chat import Anthropic_Chat
from vanna.cohere.cohere_chat import Cohere_Chat
from vanna.google import GoogleGeminiChat
from vanna.mistral.mistral import Mistral
from vanna.openai.openai_chat import OpenAI_Chat
from vanna.remote import VannaDefault
from vanna.vannadb.vannadb_vector import VannaDB_VectorStore

# try:
#     print("Trying to load .env")
#     from dotenv import load_dotenv
#     load_dotenv()
# except Exception as e:
#     print(f"Failed to load .env {e}")
#     pass

# MY_VANNA_MODEL = 'chinook'
# ANTHROPIC_Model = 'claude-3-sonnet-20240229'
# MY_VANNA_API_KEY = os.environ['VANNA_API_KEY']
# OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
# MISTRAL_API_KEY = os.environ['MISTRAL_API_KEY']
# ANTHROPIC_API_KEY = os.environ['ANTHROPIC_API_KEY']
# SNOWFLAKE_ACCOUNT = os.environ['SNOWFLAKE_ACCOUNT']
# SNOWFLAKE_USERNAME = os.environ['SNOWFLAKE_USERNAME']
# SNOWFLAKE_PASSWORD = os.environ['SNOWFLAKE_PASSWORD']
# AZURE_SEARCH_API_KEY = os.environ['AZURE_SEARCH_API_KEY']



from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
# from vanna.openai.openai_chat import OpenAI_Chat

from vanna.ollama import Ollama

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

config = {
    "path": "./tests/chroma_db", #向量数据库存储路径
    "model": "qwen3-coder:480b-cloud",  # 本地 Ollama 模型名称
    "ollama_host": "http://192.168.3.189:11434",  # Ollama 服务地址
    "persist_directory": "./chroma_db",  # 向量数据库存储路径-----这个参数没有发挥作用
    "options": {"temperature": 0.3}  # 控制生成随机性
}

vn = MyVanna(config=config)

vn.connect_to_sqlite(r'C:\Tiigee\git_repositories\vanna\tests\database\Chinook.sqlite')

def test_vn_chroma():
    existing_training_data = vn.get_training_data()
    if len(existing_training_data) > 0:
        for _, training_data in existing_training_data.iterrows():
            vn.remove_training_data(training_data['id'])

    df_ddl = vn.run_sql("SELECT type, sql FROM sqlite_master WHERE sql is not null")

    for ddl in df_ddl['sql'].to_list():
        vn.train(ddl=ddl)

    # sql = vn.generate_sql("What are the top 7 customers by sales?")
    sql = vn.generate_sql("What are the top 10 customers by sales?")

    print("*******************************begin\n")
    print(sql)
    print("*******************************end\n")

    df = vn.run_sql(sql)
    assert len(df) == 10
    
if __name__ == "__main__":
    test_vn_chroma()


# class VannaNumResults(ChromaDB_VectorStore, OpenAI_Chat):
#     def __init__(self, config=None):
#         ChromaDB_VectorStore.__init__(self, config=config)
#         OpenAI_Chat.__init__(self, config=config)

# vn_chroma_n_results = MyVanna(config={'model': 'gpt-3.5-turbo', 'n_results': 1})
# vn_chroma_n_results_ddl = MyVanna(config={'model': 'gpt-3.5-turbo', 'n_results_ddl': 2})
# vn_chroma_n_results_sql = MyVanna(config={'model': 'gpt-3.5-turbo', 'n_results_sql': 3})
# vn_chroma_n_results_documentation = MyVanna(config={'model': 'gpt-3.5-turbo', 'n_results_documentation': 4})

# def test_n_results():
#     for i in range(1, 10):
#         vn.train(question=f"What are the total sales for customer {i}?", sql=f"SELECT SUM(sales) FROM example_sales WHERE customer_id = {i}")

#     for i in range(1, 10):
#         vn.train(documentation=f"Sample documentation {i}")

#     question = "Whare are the top 5 customers by sales?"
#     assert len(vn_chroma_n_results.get_related_ddl(question)) == 1
#     assert len(vn_chroma_n_results.get_related_documentation(question)) == 1
#     assert len(vn_chroma_n_results.get_similar_question_sql(question)) == 1

#     assert len(vn_chroma_n_results_ddl.get_related_ddl(question)) == 2
#     assert len(vn_chroma_n_results_ddl.get_related_documentation(question)) != 2
#     assert len(vn_chroma_n_results_ddl.get_similar_question_sql(question)) != 2

#     assert len(vn_chroma_n_results_sql.get_related_ddl(question)) != 3
#     assert len(vn_chroma_n_results_sql.get_related_documentation(question)) != 3
#     assert len(vn_chroma_n_results_sql.get_similar_question_sql(question)) == 3

#     assert len(vn_chroma_n_results_documentation.get_related_ddl(question)) != 4
#     assert len(vn_chroma_n_results_documentation.get_related_documentation(question)) == 4
#     assert len(vn_chroma_n_results_documentation.get_similar_question_sql(question)) != 4
