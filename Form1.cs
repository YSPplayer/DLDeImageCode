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
            // ����һ��200x100�Ŀհ�ͼƬ
            Bitmap bitmap = new Bitmap(200, 100);
            Graphics g = Graphics.FromImage(bitmap);
            g.Clear(Color.White); // ���ñ���Ϊ��ɫ

            Random random = new Random();
            Font font = new Font("Arial", 24, FontStyle.Bold); // ��������

            for (int i = 0; i < 4; i++)
            {
                string number = random.Next(0, 10).ToString(); // ����һ��0-9������
                strings.Add(number);
                Brush brush = new SolidBrush(RandomColor(random)); // �����ɫ

                // �����ַ��ĳߴ�
                SizeF size = g.MeasureString(number, font);

                // ��ת�Ƕ�
                float angle = random.Next(-40, 40);

                // �����ַ����Ƶ���ʼλ��
                float x = i * (200 / 4) + (200 / 8) - (size.Width / 2);
                float y = 50 - (size.Height / 2);

                // Ӧ����ת
                g.TranslateTransform(x + size.Width / 2, y + size.Height / 2);
                g.RotateTransform(angle);
                g.TranslateTransform(-size.Width / 2, -size.Height / 2);

                // �����ַ�
                g.DrawString(number, font, brush, new PointF(0, 0));

                // ����ͼ�α任
                g.ResetTransform();
            }
            // ��Ӻ��߻����ߵ��������
            for (int j = 0; j < 10; j++)
            {
                int x1 = random.Next(bitmap.Width);
                int y1 = random.Next(bitmap.Height);
                int x2 = random.Next(bitmap.Width);
                int y2 = random.Next(bitmap.Height);
                g.DrawLine(new Pen(RandomColor(random), 1), new Point(x1, y1), new Point(x2, y2));
            }
            string imagePath = Path.Combine(path, $"{strings[0]}{strings[1]}{strings[2]}{strings[3]}.jpg");
            // ����ͼƬ������
            bitmap.Save(imagePath);

            // ������Դ
            g.Dispose();
            bitmap.Dispose();
        }

        // ���������ɫ�ķ���
        static Color RandomColor(Random random)
        {
            return Color.FromArgb(random.Next(256), random.Next(256), random.Next(256));
        }
    }
}
