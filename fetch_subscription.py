import stripe

# Imposta la chiave segreta di Stripe
stripe.api_key = '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'

# Recupera la lista degli abbonati
subscribers = stripe.Subscription.list()

# Stampa le informazioni sugli abbonati
for subscription in subscribers.data:
    customer_id = subscription.customer
    customer = stripe.Customer.retrieve(customer_id)
    plan = subscription.plan
    product = stripe.Product.retrieve(plan.product)

    print("Email dell'utente abbonato:", customer.email)
    print("Stato dell'abbonamento:", subscription.status)
    print("Prodotto:", product.name)  # Stampa il nome del prodotto

    #print(subscription)
