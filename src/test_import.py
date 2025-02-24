# src/test_import.py

from utils import setup_logging
import logging

def main():
    setup_logging(log_file='logs/test_import.log')
    logging.info("日志设置成功。")

if __name__ == "__main__":
    main()