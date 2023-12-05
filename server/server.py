from fastapi import FastAPI
from fastapi.responses import JSONResponse

import pandas as pd
from endpoint_util import EndpointUtil


local_folder_path = '/data/'
endpoint_name= "perfume-recomand-ml2-last1"
bucket_name = "sagemaker-gacheon-ml2-team1"
endpoint = EndpointUtil(bucket_name, endpoint_name, local_folder_path)


app = FastAPI()

@app.get("/")
async def hello():
    return {"hello":"world"}

@app.get('/get_perfume_info')
async def get_movie_info():
    perfume_df = pd.read_csv('noon_perfumes_dataset.csv')
    perfume_df = perfume_df.drop(['Unnamed: 0'], axis=1)
    print(perfume_df)
    return JSONResponse(content=perfume_df.to_dict(orient='records'))


@app.get("/{user_id}")
async def root(user_id: str):
    pred_df = endpoint.call(int(user_id))
    pred_df = pred_df.to_dict(orient='records')
    print(pred_df)
    return JSONResponse(content=pred_df)

@app.get('/{satisfaction}/{buyer_gender}/{buyer_age}/{first_note}/{f_note_degree}/{second_note}/{s_note_degree}')
async def get_perfume_match(
    satisfaction: str, buyer_gender: str, buyer_age: str,
    first_note: str, f_note_degree: str, second_note: str, s_note_degree: str
):
    pred_df = endpoint.call_detail(str(satisfaction), str(buyer_gender), int(buyer_age), str(first_note), float(f_note_degree), str(second_note), float(s_note_degree))
    pred_df = pred_df.to_dict(orient='records')
    return JSONResponse(content=pred_df)
    
    

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("server:app", host='0.0.0.0', port=8000, workers=1)  # reload=False 권장
