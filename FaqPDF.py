from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

doc = SimpleDocTemplate("Timely_faq.pdf", pagesize=letter)
styles = getSampleStyleSheet()
content = []

faq_text = [
    ("Timely - Customer FAQ & Policy Guide", "Title"),

    ("1. Return Policy", "Heading2"),
    ("Customers can return any product within 7 days of delivery. "
     "The product must be unused, in original packaging with all accessories. "
     "To initiate a return contact support at support@timely.pk or call 0327-0337903. "
     "Refunds are processed within 5-7 business days after receiving the returned item.", "Normal"),

    ("2. Warranty Policy", "Heading2"),
    ("All Watches and Accessories at Timely come with a 1 year official brand warranty. "
     "Rolex products have 1 year warranty. SKMEI products have 1 year warranty. "
     "Sveston products have 2 years warranty. Casio products have 1 year warranty. "
     "Warranty does not cover physical damage, water damage, or unauthorized repairs.", "Normal"),

    ("3. Delivery Information", "Heading2"),
    ("Timely delivers across Pakistan. Delivery times are as follows: "
     "Lahore: 1-2 business days. Karachi and Islamabad: 2-3 business days. "
     "Other cities: 3-5 business days. "
     "Delivery charges are Rs 200 for orders below Rs 5000. "
     "Free delivery on orders above Rs 5000. "
     "Express delivery available in Lahore for Rs 300.", "Normal"),

    ("4. Payment Methods", "Heading2"),
    ("Timely accepts the following payment methods: "
     "Cash on Delivery (COD) available nationwide. "
     "Easypaisa and JazzCash mobile payments accepted. "
     "Visa and Mastercard debit and credit cards accepted. "
     "Bank transfer available for orders above Rs 10000.", "Normal"),

    ("5. Products We Sell", "Heading2"),
    ("Timely specializes in Watches and Accessories including: "
     "Smart Watches from Apple, Samsung Galaxy Watch, Huawei, Xiaomi, and Garmin. "
     "Vintage and Classic Watches from Casio, Seiko, Orient, and Fossil. "
     "Luxury Watch Replicas and fashion watches for men and women. "
     "Watch Accessories including leather straps, metal bands, silicon straps, and watch winders. "
     "Watch Storage including luxury watch boxes, travel cases, and display stands. "
     "Wallets including leather wallets, card holders, and money clips from local and imported brands. "
     "Gift Sets including watch and wallet combos, perfect for gifting. "
     "Maintenance products including watch cleaning kits and screen protectors.", "Normal"),

    ("6. Order Tracking", "Heading2"),
    ("Customers can track their orders by visiting timely.pk/track "
     "and entering their order ID received via SMS or email. "
     "Order updates are sent via SMS at every stage. "
     "For tracking issues contact support at 0327-0337903.", "Normal"),

    ("7. Damaged or Wrong Product", "Heading2"),
    ("If you receive a damaged or wrong product, contact us within 24 hours of delivery. "
     "Send photos of the product to support@timely.pk. "
     "We will arrange a free pickup and send the correct product within 3 business days. "
     "Full refund is also available if replacement is not preferred.", "Normal"),

    ("8. Cancellation Policy", "Heading2"),
    ("Orders can be cancelled within 2 hours of placing them. "
     "To cancel call 0327-0337903 or email support@timely.pk. "
     "Orders that have already been dispatched cannot be cancelled. "
     "COD orders refused at delivery will incur a Rs 200 penalty fee.", "Normal"),
]

for text, style in faq_text:
    content.append(Paragraph(text, styles[style]))
    content.append(Spacer(1, 12))

doc.build(content)
print("Timely_faq.pdf created successfully!")