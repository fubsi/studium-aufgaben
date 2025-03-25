import jinja2

def print_mails():
    environment = jinja2.Environment()
    template = environment.from_string("Hello {{name}}!\n"
                                       "How are you doing?\n"
                                       "Greetings")
    
    for name in ["Luke", "Anakin", "Darth"]:
        print(template.render(name=name))

def print_text():
    environment = jinja2.Environment()
    template = environment.from_string("{% for text in texts %} {{text}} {% endfor %}")
    texts = ["Luke is Jedi", "Anakin is Sith", "Darth is Vader", "Han is Warrior"]
    print(template.render(texts=texts))

print_text()