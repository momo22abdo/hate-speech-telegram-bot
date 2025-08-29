# Debug version of main.py to identify startup issues

import logging
import sys
import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters

# Set up logging first
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),  # Ensure logs go to console
        logging.FileHandler('bot.log')  # Also save to file
    ]
)
logger = logging.getLogger(__name__)

print("🔧 Starting bot initialization...")
logger.info("🔧 Starting bot initialization...")

# Check if config file exists and can be imported
try:
    print("📁 Checking config file...")
    from config import TELEGRAM_BOT_TOKEN

    if TELEGRAM_BOT_TOKEN:
        print("✅ Bot token loaded successfully")
        logger.info("✅ Bot token loaded successfully")
    else:
        print("❌ Bot token is empty!")
        logger.error("❌ Bot token is empty!")
        sys.exit(1)
except ImportError as e:
    print(f"❌ Cannot import config: {e}")
    logger.error(f"❌ Cannot import config: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error loading config: {e}")
    logger.error(f"❌ Error loading config: {e}")
    sys.exit(1)

# Check if utils modules can be imported
try:
    print("📚 Checking utils modules...")
    from utils.classification import classify_text
    from utils.explain import explain_prediction
    from utils.stats import update_user_stats, generate_stats_report

    print("✅ Utils modules loaded successfully")
    logger.info("✅ Utils modules loaded successfully")
except ImportError as e:
    print(f"❌ Cannot import utils: {e}")
    logger.error(f"❌ Cannot import utils: {e}")
    sys.exit(1)

# Check if models exist
model_path = os.path.join('models', 'classifier.joblib')
vectorizer_path = os.path.join('models', 'vectorizer.joblib')

if not os.path.exists(model_path):
    print(f"❌ Model file not found: {model_path}")
    logger.error(f"❌ Model file not found: {model_path}")
    print("💡 Please run train_model.py first to create the model files")
    sys.exit(1)

if not os.path.exists(vectorizer_path):
    print(f"❌ Vectorizer file not found: {vectorizer_path}")
    logger.error(f"❌ Vectorizer file not found: {vectorizer_path}")
    print("💡 Please run train_model.py first to create the model files")
    sys.exit(1)

print("✅ Model files found")
logger.info("✅ Model files found")


async def start(update: Update, context):
    """Handle the /start command."""
    logger.info("Received /start command from user %s", update.message.from_user.id)
    print(f"📱 Received /start from user {update.message.from_user.id}")
    await update.message.reply_text(
        "🤖 **Welcome to the Enhanced Hate Speech Detection Bot!**\n\n"
        "📝 Send any text, and I'll analyze it for:\n"
        "• Hate Speech\n"
        "• Offensive Language\n"
        "• Self-harm content\n"
        "• Threats\n\n"
        "📊 Use /stats to see your classification statistics.\n"
        "🔍 Click 'Explain Decision' to understand why text was classified a certain way.",
        parse_mode='Markdown'
    )


async def classify_handler(update: Update, context):
    """Handle text messages: classify the text."""
    text = update.message.text
    user_id = update.message.from_user.id
    logger.info("Received text '%s' from user %s", text, user_id)
    print(f"📝 Processing text: '{text}' from user {user_id}")

    try:
        # Classify the text
        prediction = classify_text(text)
        print(f"🎯 Classification result: {prediction}")

        # Update user stats
        base_label = prediction['label'].split(' (')[0]
        update_user_stats(user_id, base_label)

        # Store for explain functionality
        context.user_data['last_text'] = text
        context.user_data['last_pred'] = prediction

        # Simple response for debugging
        response = f"🎯 Classification: {prediction['label']}\n📊 Confidence: {prediction['confidence']}%"

        # Add flags if present
        flags = prediction.get('flags', [])
        if flags:
            response += f"\n🏷️ Flags: {', '.join(flags)}"

        # Inline buttons
        keyboard = [
            [
                InlineKeyboardButton("📊 Stats", callback_data='stats'),
                InlineKeyboardButton("🔍 Explain", callback_data='explain')
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(response, reply_markup=reply_markup)
        print("✅ Response sent successfully")

    except Exception as e:
        error_msg = f"❌ Error processing text: {str(e)}"
        print(error_msg)
        logger.error(error_msg)
        await update.message.reply_text("❌ Sorry, an error occurred while processing your text.")


async def stats_handler(update: Update, context):
    """Handle the /stats command."""
    user_id = update.message.from_user.id
    logger.info("Received /stats command from user %s", user_id)
    print(f"📊 Stats request from user {user_id}")

    try:
        path = generate_stats_report(user_id)
        if path:
            await update.message.reply_photo(photo=open(path, 'rb'))
            print("✅ Stats chart sent")
        else:
            await update.message.reply_text("📊 No statistics available yet. Analyze some texts first!")
            print("ℹ️ No stats available for user")
    except Exception as e:
        error_msg = f"❌ Error generating stats: {str(e)}"
        print(error_msg)
        logger.error(error_msg)


async def callback_handler(update: Update, context):
    """Handle inline button callbacks."""
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()
    logger.info("Received callback '%s' from user %s", query.data, user_id)
    print(f"🔘 Button pressed: {query.data} by user {user_id}")

    if query.data == 'stats':
        try:
            path = generate_stats_report(user_id)
            if path:
                await query.message.reply_photo(photo=open(path, 'rb'))
            else:
                await query.message.reply_text("📊 No statistics available yet.")
        except Exception as e:
            logger.error("Error in stats callback: %s", str(e))
            await query.message.reply_text("❌ Error generating stats.")

    elif query.data == 'explain':
        try:
            last_text = context.user_data.get('last_text')
            last_pred = context.user_data.get('last_pred')
            if last_text and last_pred:
                explanation = explain_prediction(last_text, last_pred)
                await query.message.reply_text(f"🔍 **Explanation:** {explanation}", parse_mode='Markdown')
            else:
                await query.message.reply_text("❓ No recent prediction to explain.")
        except Exception as e:
            logger.error("Error in explain callback: %s", str(e))
            await query.message.reply_text("❌ Error generating explanation.")


def main():
    """Start the bot with comprehensive error handling."""
    try:
        print("🔧 Creating application...")
        logger.info("🔧 Creating application...")

        application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
        print("✅ Application created successfully")
        logger.info("✅ Application created successfully")

        # Add handlers
        print("🔗 Adding handlers...")
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("stats", stats_handler))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, classify_handler))
        application.add_handler(CallbackQueryHandler(callback_handler))
        print("✅ Handlers added successfully")
        logger.info("✅ Handlers added successfully")

        # Start polling
        print("🚀 Starting bot polling...")
        print("✅ Bot is now running! Send a message to test it.")
        print("🛑 Press Ctrl+C to stop the bot.")
        logger.info("🚀 Bot polling started")

        application.run_polling()

    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user (Ctrl+C)")
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        error_msg = f"❌ Critical error starting bot: {str(e)}"
        print(error_msg)
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()