call activate python_37
call set KERASTUNER_TUNER_ID=chief
call set KERASTUNER_ORACLE_IP=localhost
call set KERASTUNER_ORACLE_PORT=8000
call python training.py