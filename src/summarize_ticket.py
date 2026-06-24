def summarize_ticket(ticket_text):

    sentences = ticket_text.split(".")

    if len(sentences) > 0:
        return sentences[0].strip()

    return ticket_text


if __name__ == "__main__":

    ticket = """
    Unable to log into my account. I tried resetting my password
    multiple times but never received the reset email. This issue
    is preventing me from accessing important documents.
    """

    print("Summary:")
    print(summarize_ticket(ticket))