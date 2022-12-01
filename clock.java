package task;

import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.event.ActionListener;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Locale;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JTextField;

public class clock extends JFrame {

	class MyRunnable implements Runnable {
		public void run() {
			Calendar calendar = Calendar.getInstance();
			String pattern = "a hh:mm:ss";
			SimpleDateFormat f = new SimpleDateFormat(pattern, Locale.KOREA);
			time.setText(f.format(calendar.getTime()));
		}
	}

	JButton start, stop;
	JTextField time;

	clock() {
		setTitle("시계");
		setLayout(new BorderLayout(10, 10));

		showNorth();
		showSouth();

		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setSize(300, 105);
		setVisible(true);
	}

	void showNorth() {
		JPanel p1 = new JPanel();
		time = new JTextField(25);

		time.setEnabled(false);

		p1.add(time);

		add(p1, BorderLayout.NORTH);
	}

	void showSouth() {
		JPanel sp = new JPanel(new FlowLayout(FlowLayout.CENTER, 10, 10));

		start = new JButton("시작");

		ActionListener listener1 = e -> {
			if (e.getSource() == start) {
				Thread t = new Thread(new MyRunnable());
				t.start();

			}
		};

		start.addActionListener(listener1);

		stop = new JButton("정지");

		ActionListener listener2 = e -> {
			if (e.getSource() == stop) {
				Calendar calendar = Calendar.getInstance();
				String pattern = "a hh:mm:ss";
				SimpleDateFormat f = new SimpleDateFormat(pattern, Locale.KOREA);
				time.setText(f.format(calendar.getTime()));
			}
		};

		stop.addActionListener(listener2);

		sp.add(start);
		sp.add(stop);

		add(sp, BorderLayout.SOUTH);

	}

	public static void main(String[] args) {
		new clock();
	}
}
