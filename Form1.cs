using Microsoft.VisualBasic;
using System.Drawing;
using System.Drawing.Drawing2D;

namespace DLDeImageCode
{
    public partial class Form1 : Form
    {
        List<string> strings = new List<string>();
        public Form1()
        {
            InitializeComponent();
            string folderPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Image");
            for (int i = 0; i < 1; i++)
            {
                folderPath = Path.Combine(folderPath, i.ToString());
                if (!Directory.Exists(folderPath))
                {
                    Directory.CreateDirectory(folderPath);
                }
                for (int j = 0; j < 1; j++)
                {
                    CreateImageCode(folderPath);
                }

            }

        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        void CreateImageCode(string path) 
        {
            strings.Clear();
            // 创建一个200x100的空白图片
            Bitmap bitmap = new Bitmap(200, 100);
            Graphics g = Graphics.FromImage(bitmap);
            g.Clear(Color.White); // 设置背景为白色

            Random random = new Random();
            Font font = new Font("Arial", 24, FontStyle.Bold); // 设置字体

            for (int i = 0; i < 4; i++)
            {
                string number = random.Next(0, 10).ToString(); // 生成一个0-9的数字
                strings.Add(number);
                Brush brush = new SolidBrush(RandomColor(random)); // 随机颜色

                // 计算字符的尺寸
                SizeF size = g.MeasureString(number, font);

                // 旋转角度
                float angle = random.Next(-40, 40);

                // 计算字符绘制的起始位置
                float x = i * (200 / 4) + (200 / 8) - (size.Width / 2);
                float y = 50 - (size.Height / 2);

                // 应用旋转
                g.TranslateTransform(x + size.Width / 2, y + size.Height / 2);
                g.RotateTransform(angle);
                g.TranslateTransform(-size.Width / 2, -size.Height / 2);

                // 绘制字符
                g.DrawString(number, font, brush, new PointF(0, 0));

                // 重置图形变换
                g.ResetTransform();
            }
            // 添加横线或竖线的随机噪音
            for (int j = 0; j < 10; j++)
            {
                int x1 = random.Next(bitmap.Width);
                int y1 = random.Next(bitmap.Height);
                int x2 = random.Next(bitmap.Width);
                int y2 = random.Next(bitmap.Height);
                g.DrawLine(new Pen(RandomColor(random), 1), new Point(x1, y1), new Point(x2, y2));
            }
            string imagePath = Path.Combine(path, $"{strings[0]}{strings[1]}{strings[2]}{strings[3]}.jpg");
            // 保存图片到磁盘
            bitmap.Save(imagePath);

            // 清理资源
            g.Dispose();
            bitmap.Dispose();
        }

        // 生成随机颜色的方法
        static Color RandomColor(Random random)
        {
            return Color.FromArgb(random.Next(256), random.Next(256), random.Next(256));
        }
    }
}
