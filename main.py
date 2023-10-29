from api import app as app

if __name__ == "__main__":

    # app.run(host="127.0.0.1", port=3000)
    app.run(host="0.0.0.0", port=80, debug=True)


#  /home/enes/anaconda3/envs/hs/bin/python main.py