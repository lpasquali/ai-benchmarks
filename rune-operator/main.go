package main

import (
	"flag"
	"os"

	"k8s.io/apimachinery/pkg/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/klog/v2"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	"sigs.k8s.io/controller-runtime/pkg/metrics/server"

	benchv1alpha1 "github.com/lpasquali/ai-benchmarks/rune-operator/api/v1alpha1"
	"github.com/lpasquali/ai-benchmarks/rune-operator/controllers"
	"github.com/lpasquali/ai-benchmarks/rune-operator/internal/metrics"
	"github.com/lpasquali/ai-benchmarks/rune-operator/internal/telemetry"
)

func main() {
	var metricsAddr string
	var probeAddr string
	var enableLeaderElection bool

	flag.StringVar(&metricsAddr, "metrics-bind-address", ":8080", "The address the metric endpoint binds to.")
	flag.StringVar(&probeAddr, "health-probe-bind-address", ":8081", "The address the probe endpoint binds to.")
	flag.BoolVar(&enableLeaderElection, "leader-elect", true, "Enable leader election for controller manager.")
	opts := zap.Options{Development: false}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()

	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&opts)))

	s := runtimeScheme()

	shutdownTelemetry := telemetry.SetupOTel("rune-operator")
	defer shutdownTelemetry()

	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme:                 s,
		Metrics:                server.Options{BindAddress: metricsAddr},
		HealthProbeBindAddress: probeAddr,
		LeaderElection:         enableLeaderElection,
		LeaderElectionID:       "rune-operator.bench.rune.ai",
	})
	if err != nil {
		klog.ErrorS(err, "unable to start manager")
		os.Exit(1)
	}

	metrics.Register()

	if err = (&controllers.RuneBenchmarkReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
		Recorder: mgr.GetEventRecorderFor("rune-benchmark-controller"),
	}).SetupWithManager(mgr); err != nil {
		klog.ErrorS(err, "unable to create controller", "controller", "RuneBenchmark")
		os.Exit(1)
	}

	if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		klog.ErrorS(err, "unable to set up health check")
		os.Exit(1)
	}
	if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
		klog.ErrorS(err, "unable to set up ready check")
		os.Exit(1)
	}

	klog.InfoS("starting manager")
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		klog.ErrorS(err, "problem running manager")
		os.Exit(1)
	}
}

func runtimeScheme() *runtime.Scheme {
	s := runtime.NewScheme()
	_ = clientgoscheme.AddToScheme(s)
	_ = benchv1alpha1.AddToScheme(s)
	return s
}
